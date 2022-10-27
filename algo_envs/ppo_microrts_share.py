import sys,os,time
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch 
import torch.nn as nn
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from collections import deque
import numpy as np
from torch.distributions.categorical import Categorical
import algo_envs.algo_base as AlgoBase
from types import SimpleNamespace
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

"""
Environments' objects for training, including
    map_size: observation space size map
    conv_linear: neural numbers from multi-dimentional to 1-dimentional.
    map_path: path of map file
    num_steps: running timestpes for each episode
    max_version: maximum number of episodes

Example:
    >>> map_size = train_envs[current_env_name].map_size
    >>> map_path = train_envs[current_env_name].map_path
    >>> max_version = train_envs[current_env_name].max_version

"""

train_envs = {
    '10_10':SimpleNamespace(**{'map_size': 10*10,'conv_linear':3*3,'map_path':'maps/10x10/basesWorkers10x10.xml','num_steps':512,'max_version':int(5e4)}),
    '16_16':SimpleNamespace(**{'map_size': 16*16,'conv_linear':6*6,'map_path':'maps/16x16/basesWorkers16x16.xml','num_steps':512,'max_version':int(3e3)}),
    '24_24':SimpleNamespace(**{'map_size': 24*24,'conv_linear':6*6,'map_path':'maps/16x16/basesWorkers16x16.xml','num_steps':512,'max_version':int(5e4)}),
}

# current environment name
current_env_name = '10_10'

#training parameters
train_config = dict()

# gae lambda
train_config['gae_lambda'] = 0.95 

# discount factor
train_config['gamma'] = 0.99 

# policy gradient loss clip
train_config['clip_coef'] = 0.2 

# ratio upper bound
train_config['max_clip_coef'] = 4 

# weight of entropy
train_config['ent_coef'] = 0.01 

# weight of value loss
train_config['vf_coef'] = 1 

# learning rate
train_config['learning_rate'] = 2.5e-4 

# weight of ratio
train_config['ratio_coef'] = 0.5 

# max gradient normal of value
train_config['grad_norm'] = 0.5 

# minibatch size, which should divisible by an epdsode length
train_config['mini_batch_size'] = 128 

# policy gradient type: 0 is discrete, 1 is continues, 2 is mixed and 3 is mixed policy gradient loss
train_config['pg_loss_type'] = 0 

# whether clip value loss
train_config['enable_clip_max'] = True 

# whether to decay ratio 
train_config['enable_ratio_decay'] = False 

# whether to decay entropy coefficient
train_config['enable_entropy_decay'] = False 

# whether to decay learning rate
train_config['enable_lr_decay'] = True 

# whether to clip gradient norm
train_config['enable_grad_norm'] = False 

# whether to normalize advantage function
train_config['enable_adv_norm'] = True 

# whether to use mini batch
train_config['enable_mini_batch'] = False 

# the number of environments
train_config['num_envs'] = 1 

# action space of environments
train_config['action_shape'] = [train_envs[current_env_name].map_size, 6, 4, 4, 4, 4, 7, 49] 

# whether to use GPU to training
train_config['use_gpu'] = False 

# for tensorboard naming
train_config['tensorboard_comment'] = 'todo'

class PPOMicroRTSShareNet(AlgoBase.AlgoBaseNet):
    """Policy class used with PPO

    Example:
    >>> current_env_name = '10_10' # environment name
    >>> train_net = PPOMicroRTSShareNet()
    >>> state = torch.randn(1,10,10,27) # one state dimention
    >>> action = torch.randn(1,8) # one action dimention
    >>> train_net.get_distris(state)
    >>> distris, value = train_net(state)
    """
    def __init__(self):
        super(PPOMicroRTSShareNet,self).__init__()

        self.policy_network = nn.Sequential(
            AlgoBase.layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            AlgoBase.layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            AlgoBase.layer_init(nn.Linear(32 * train_envs[current_env_name].conv_linear, 256)),
            nn.ReLU(),
        )

        self.policy_unit = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(256, train_envs[current_env_name].map_size), std=0.01),
            )
        self.policy_type = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(256, 6), std=0.01),
            )
        self.policy_move = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(256, 4), std=0.01),
            )
        self.policy_harvest = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(256, 4), std=0.01)
            )
        self.policy_return = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(256, 4), std=0.01),
            )
        self.policy_produce = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(256, 4), std=0.01),
            )
        self.policy_produce_type = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(256, 7), std=0.01),
            )
        self.policy_attack = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(256, 49), std=0.01),
            )
        
        self.value = nn.Sequential(
                AlgoBase.layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
                nn.ReLU(),
                nn.Flatten(),
                AlgoBase.layer_init(nn.Linear(32 * train_envs[current_env_name].conv_linear, 256)),
                nn.ReLU(), 
                AlgoBase.layer_init(nn.Linear(256, 1), std=1)
            )
                
    def get_distris(self,states):
        """
        Calculate the distributions of states

        Args:
            states

        Return:
            distribution of states

        Example:
            >>> current_env_name = '10_10' # environment name
            >>> train_net = PPOMicroRTSShareNet()
            >>> state = torch.randn(1,10,10,27) # one state dimention
            >>> action = torch.randn(1,8) # one action dimention
            >>> train_net.get_distris(state)
        """
        # Moving last convolution channel shape to the second dimention 
        states = states.permute((0, 3, 1, 2))
        policy_network = self.policy_network(states)
            
        unit_distris = AlgoBase.MaskedCategorical(self.policy_unit(policy_network))
        type_distris = AlgoBase.MaskedCategorical(self.policy_type(policy_network))
        move_distris = AlgoBase.MaskedCategorical(self.policy_move(policy_network))
        harvest_distris = AlgoBase.MaskedCategorical(self.policy_harvest(policy_network))
        return_distris = AlgoBase.MaskedCategorical(self.policy_return(policy_network))
        produce_distris = AlgoBase.MaskedCategorical(self.policy_produce(policy_network))
        produce_type_distris = AlgoBase.MaskedCategorical(self.policy_produce_type(policy_network))
        attack_distris = AlgoBase.MaskedCategorical(self.policy_attack(policy_network))

        return [unit_distris,type_distris,move_distris,harvest_distris,return_distris,produce_distris,produce_type_distris,attack_distris]

    def forward(self, states):
        """
        Calculate state values, probability distributions of each state

        Args:
            states

        Return:
            probability distributions and state values 

        Example:
            >>> current_env_name = '10_10' # environment name
            >>> train_net = PPOMicroRTSShareNet()
            >>> state = torch.randn(1,10,10,27) # one state dimention
            >>> distris, value = train_net(state)
        """
        distris = self.get_distris(states)
        value = self.get_value(states)
        return distris,value
    
    def get_value(self,states):
        """
        Calculate state value of each state

        Args:
            states
        
        Return:
            state-values

        Example:
            >>> current_env_name = '10_10' # environment name
            >>> train_net = PPOMicroRTSShareNet()
            >>> state = torch.randn(1,10,10,27) # one state dimention
            >>> train_net.get_value(state)
        """

        # Moving last convolution channel shape to the second dimention 
        states = states.permute((0, 3, 1, 2))
        return self.value(states)
    
class PPOMicroRTSShareUtils(AlgoBase.AlgoBaseUtils):
    pass
        
class PPOMicroRTSShareAgent(AlgoBase.AlgoBaseAgent):
    """
    Agent class used with PPO, allowing collect data and evaluate agents.

    Args:
        sample_net: policy network (default: PPOMicroRTSShareNet)
        model_dict: a dict of model configuration
        is_checker: if "True", then evaluating the agent through running 1024 timesteps with 
        the highest probility of action, else collecting the training data.

    Example:
        >>> train_net = PPOMicroRTSShareNet()
        >>> # Collecting training data
        >>> sample_agent = PPOMicroRTSShareAgent(train_net,model_dict,is_checker=False)
        >>> transition = sample_agent.sample_env()
        >>> # Evaluating agent 
        >>> check_agent = PPOMujocoNormalShareAgent(train_net,model_dict,is_checker=True)
        >>> info = check_agent.check_env()
        >>> print(info['sum_rewards'], info['mean_entropys'], info['mean_log_probs'], ['mean_win_rates'])

    """

    def __init__(self,sample_net:PPOMicroRTSShareNet,model_dict,is_checker=False):
        super(PPOMicroRTSShareAgent,self).__init__()
        self.sample_net = sample_net
        self.model_dict = model_dict
        self.num_envs = train_config['num_envs']
        self.num_check_envs = 8
        self.num_steps = train_envs[current_env_name].num_steps
        self.action_shape = train_config['action_shape']

        if not is_checker:
            self.env = MicroRTSVecEnv(
                num_envs=self.num_envs,
                max_steps=5000,
                ai2s=[microrts_ai.coacAI for _ in range(self.num_envs)],
                map_path=train_envs[current_env_name].map_path,
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
            )
        else:
            self.env = self.env = MicroRTSVecEnv(
                num_envs=self.num_check_envs,
                max_steps=5000,
                ai2s=[microrts_ai.coacAI for _ in range(self.num_check_envs)],
                map_path=train_envs[current_env_name].map_path,
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
            )
            self.num_steps = 1024
            self.out_comes = deque( maxlen= 1000)
            print("PPOMicroRTSShare check map is",current_env_name)
        self.obs = self.env.reset()
        
    def get_comment_info(self):
        return current_env_name + "_" + train_config['tensorboard_comment']
            
    def sample_env(self):
        """
        Collect training data

        Example:
            >>> train_net = PPOMicroRTSShareNet()
            >>> sample_agent = PPOMicroRTSShareAgent(train_net,model_dict,is_checker=False)
            >>> transition = sample_agent.sample_env()
        """
        exps=[[] for _ in range(self.num_envs)]
        
        for step in range(self.num_steps):
            unit_mask = np.array(self.env.vec_client.getUnitLocationMasks()).reshape(self.num_envs, -1)
  
            action,mask,log_prob=self.get_sample_actions(self.obs, unit_mask)
            next_obs, rs, done_n, _ = self.env.step(action)
            
            for i in range(self.num_envs):
                
                if done_n[i] or step == self.num_steps - 1:
                    done = True
                else:
                    done = False
                
                exps[i].append([self.obs[i],action[i],rs[i],mask[i],done,log_prob[i],self.model_dict['train_version']])
                
            self.obs=next_obs

        return exps

    def check_env(self):
        """
        Evaluate agent

        Example:
            >>> train_net = PPOMicroRTSShareNet()
            >>> check_agent = PPOMicroRTSShareAgent(train_net,model_dict,is_checker=True)
            >>> transition = sample_agent.check_env()
        """
        step_record_dict = dict()

        rewards = []
        entropys = []
        log_probs = []
        
        for _ in range(0, self.num_steps):
            #self.env.render()
            unit_masks = np.array(self.env.vec_client.getUnitLocationMasks()).reshape(self.num_check_envs, -1)

            action,entropy,log_prob = self.get_check_action(self.obs, unit_masks)
            next_obs, rs, done, infos = self.env.step(action)
            rewards.append(np.mean(rs))
            entropys.append(np.mean(entropy))
            log_probs.append(np.mean(log_prob))
                                
            for i in range(self.num_check_envs):
                if done[i]:
                    #if self.get_units_number(11, self.obs, i) > self.get_units_number(12, self.obs, i):
                    if infos[i]['raw_rewards'][0] > 0:
                        self.out_comes.append(1.0)
                    else:
                        self.out_comes.append(0.0)
                        
            self.obs=next_obs
                            
        mean_win_rates = np.mean(self.out_comes) if len(self.out_comes)>0 else 0.0
        print(mean_win_rates)
        
        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['mean_entropys'] = np.mean(entropys)
        step_record_dict['mean_log_probs'] = np.mean(log_probs)
        step_record_dict['mean_win_rates'] = mean_win_rates
    
        return step_record_dict
            
    def get_units_number(self,unit_type, bef_obs, ind_obs):
        return int(bef_obs[ind_obs][:, :, unit_type].sum())
    
    @torch.no_grad()
    def get_sample_actions(self,states, unit_masks):
        """
        Sample actions, masks and log probilities of actions

        Args:
            states
            unit_masks

        Returns:
            actions
            masks
            log_probs

        Example:
            >>> train_net = PPOMicroRTSShareNet()
            >>> sample_agent = PPOMicroRTSShareAgent(train_net,model_dict,is_checker=False)
            >>> self.env = MicroRTSVecEnv(
                num_envs= 1,
                max_steps=5000,
                ai2s=microrts_ai.coacAI,
                map_path=train_envs[current_env_name].map_path,
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
            )
            >>> unit_masks = np.array(self.env.vec_client.getUnitLocationMasks())
            >>> states = torch.randn(1,10,10,27)
            >>> action, mask, log_prob = sample_agent.get_sample_actions(states, unit_masks)
        """
        states = torch.Tensor(states)
        distris = self.sample_net.get_distris(states)
        
        unit_masks = torch.Tensor(unit_masks)
        distris[0].update_masks(unit_masks)
        
        units = distris[0].sample()
        action_components = [units]

        action_mask_list = np.array(self.env.vec_client.getUnitActionMasks(units.cpu().numpy())).reshape(len(units), -1)
        action_masks = torch.split(torch.Tensor(action_mask_list), self.action_shape[1:], dim=1) 
        
        action_components +=  [dist.update_masks(action_mask).sample() for dist , action_mask in zip(distris[1:],action_masks)]
            
        actions = torch.stack(action_components)
        masks = torch.cat((unit_masks, torch.Tensor(action_mask_list)), 1)
        log_probs = torch.stack([dist.log_prob(aciton) for dist,aciton in zip(distris,actions)])
        
        return actions.T.cpu().numpy(), masks.cpu().numpy(),log_probs.T.cpu().numpy()
    
    @torch.no_grad()
    def get_check_action(self,states, unit_masks):
        """
        Calculate actions, entropy and log probilities of states

        Args:
            states
            unit_masks
        
        Returns:
            actions
            entropy
            log_probs

        Example:
            >>> train_net = PPOMicroRTSShareNet()
            >>> check_agent = PPOMicroRTSShareAgent(train_net,model_dict,is_checker=True)
            >>> self.env = MicroRTSVecEnv(
                num_envs= 1,
                max_steps=5000,
                ai2s=microrts_ai.coacAI,
                map_path=train_envs[current_env_name].map_path,
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
            )
            >>> unit_masks = np.array(self.env.vec_client.getUnitLocationMasks())
            >>> state = torch.randn(1,10,10,27)
            >>> action, entropy, log_prob = check_agent.get_check_action(state, unit_masks)
        """
        states = torch.Tensor(states)
        distris = self.sample_net.get_distris(states)
        
        unit_masks = torch.Tensor(unit_masks)
        distris[0].update_masks(unit_masks)
        units = distris[0].argmax()
        
        action_components = [units]

        action_mask_list = np.array(self.env.vec_client.getUnitActionMasks(units.cpu().numpy())).reshape(len(units), -1)
        action_masks = torch.split(torch.Tensor(action_mask_list), self.action_shape[1:], dim=1) 
        action_components +=  [dist.update_masks(action_mask).argmax() for dist , action_mask in zip(distris[1:],action_masks)]
        
        actions = torch.stack(action_components)
        entropys = torch.stack([dist.entropy() for dist in distris])
        log_probs = torch.stack([dist.log_prob(aciton) for dist,aciton in zip(distris,actions)])
        
        return actions.T.cpu().numpy(),entropys.cpu().numpy(),log_probs.T.cpu().numpy()
    
class PPOMicroRTSShareCalculate(AlgoBase.AlgoBaseCalculate):
    """
    Training calss used with PPO

    Args:
        share_model: policy network (default: PPOMicroRTSShareNet)
        model_dict: a dict of model configuration
        calculate_index: the :math:`calculate_index`th agent for training

    Example:
        >>> train_net = PPOMicroRTSShareNet()
        >>> # Collecting training data
        >>> calculate = PPOMicroRTSShareCalculate(train_net, model_dict, calculate_index)
        >>> # samples are from transitions
        >>> calculate.begin_batch_train(samples)
        >>> for _ in range(REPEAT_TIMES):
        >>>    calculate.generate_grads()
        >>> calculate.end_batch_train()
    """
    def __init__(self,share_model:PPOMicroRTSShareNet,model_dict,calculate_index):
        super(PPOMicroRTSShareCalculate,self).__init__()
        self.model_dict = model_dict
        self.share_model = share_model
        
        self.calculate_number = self.model_dict['num_trainer']
        self.calculate_index = calculate_index
        self.train_version = 0
        
        if train_config['use_gpu'] and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_index = self.calculate_index % device_count
            self.device = torch.device('cuda',device_index)
        else:
            self.device = torch.device('cpu')
        
        self.calculate_net = PPOMicroRTSShareNet()
        self.calculate_net.to(self.device)
        #self.calculate_net.load_state_dict(self.share_model.state_dict())
    
        self.share_optim = torch.optim.Adam(params=self.share_model.parameters(), lr=train_config['learning_rate'])
        
        
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.marks_list = None
        
    def begin_batch_train(self, samples_list: list):
        """
        Store training data

        Example:
            >>> train_net = PPOMicroRTSShareNet()
            >>> calculate = PPOMicroRTSShareCalculate(train_net, model_dict, calculate_index)
            >>> calculate.begin_batch_train(samples)
        """
        samples = []
        for samples_item in samples_list:
            samples.extend(samples_item)
               
        s_states = np.array([s[0] for s in samples])
        s_actions = np.array([s[1] for s in samples])
        s_masks = np.array([s[3] for s in samples])
        s_log_probs = np.array([s[5] for s in samples])
        
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s[4] for s in samples])
        
        #s_versions = [s[6] for s in samples]
        
        self.states_list = torch.Tensor(s_states).to(self.device)
        self.actions_list = torch.Tensor(s_actions).to(self.device)
        self.old_log_probs_list = torch.Tensor(s_log_probs).to(self.device)
        self.marks_list = torch.Tensor(s_masks).to(self.device)
        self.rewards_list = s_rewards
        self.dones_list = s_dones
        
    def end_batch_train(self):
        """
        Clear training data and update learning rate

        Example:
            >>> train_net = PPOMicroRTSShareNet()
            >>> calculate = PPOMicroRTSShareCalculate(train_net, model_dict, calculate_index)
            >>> calculate.end_batch_train(samples)
        """
        
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.marks_list = None
        
        train_version = self.model_dict[self.calculate_index]
        self.decay_lr(train_version)
        
    def decay_lr(self, version):
        """
        Decrease learning rate:
        :math:`lr = lr(1- ve / max_ve )`
        where :math:`lr` is learning rate, :math:`ve` is current version and :math:`max_ve` is the highest version.
        Minimum learning rate is equal to 1e-5

        Example:
            >>> train_net = PPOMicroRTSShareNet()
            >>> calculate = PPOMicroRTSShareCalculate(train_net, model_dict, calculate_index)
            >>> calculate.decay_lr(calculate_index)

        """
        if train_config['enable_lr_decay']:
            lr_now = train_config['learning_rate'] * (1 - version*1.0 / train_envs[current_env_name].max_version)
            if lr_now <= 1e-5:
                lr_now = 1e-5
            
            if self.share_optim is not None:
                for param in self.share_optim.param_groups:
                    param['lr'] = lr_now
                                                                                        
    def generate_grads(self):
        """
        Calculate policy gradient loss and update share network parameters, where :math:`Loss = L^{pg} + c_1 * L^{VF} + C_2 * S`,
        :math: `L^{pg}` is policy gradient loss, :math: `L^{VF}` is value loss, :math: `S` is entropy loss and :math:`c_1, c_2`
        are coefficent, respectively. 

        Example:
            >>> train_net = PPOMicroRTSShareNet()
            >>> calculate = PPOMicroRTSShareCalculate(train_net, model_dict, calculate_index)
            >>> calculate.begin_batch_train(samples)
            >>> REPEAT_TIMES = 10
            >>> for_ in range(REPEAT_TIMES = 10):
            >>>     calculate.generate_grads()
            >>> calculate.end_batch_train()
        """
        gamma = train_config['gamma']
        gae_lambda = train_config['gae_lambda']
        pg_loss_type = train_config['pg_loss_type']
        grad_norm = train_config['grad_norm']
        vf_coef = train_config['vf_coef']
        mini_batch_size = train_config['mini_batch_size']
        
        train_version = self.model_dict[self.calculate_index]
        ent_coef = self.get_ent_coef(train_version)
        ratio_coef = self.get_ratio_coef(train_version)
        
        self.calculate_net.load_state_dict(self.share_model.state_dict())
        
        with torch.no_grad():
            _, _, policy_values = self.get_prob_entropy_value(self.states_list, self.actions_list.T, self.marks_list)
                        
        #start = timer()
        np_advantages,np_returns = AlgoBase.calculate_gae(policy_values.cpu().numpy().reshape(-1),self.rewards_list,self.dones_list,gamma,gae_lambda)
        #run_time = timer() - start
        #print("CPU function took %f seconds." % run_time)
        
        if train_config['enable_adv_norm']:
            np_advantages = (np_advantages - np_advantages.mean()) / (np_advantages.std() + 1e-8)
                                                    
        advantage_list = torch.Tensor(np_advantages.reshape(-1,1)).to(self.device)    
        returns_list = torch.Tensor(np_returns.reshape(-1,1)).to(self.device)
        
        if train_config['enable_mini_batch']:
            mini_batch_number = advantage_list.shape[0] // mini_batch_size
        else:
            mini_batch_number = 1
            mini_batch_size = advantage_list.shape[0]

        for i in range(mini_batch_number):
            start_index = i*mini_batch_size
            end_index = (i+1)* mini_batch_size
            
            mini_states = self.states_list[start_index:end_index]
            mini_actions = self.actions_list[start_index:end_index]
            mini_masks = self.marks_list[start_index:end_index]
            mini_old_log_probs = self.old_log_probs_list[start_index:end_index]
            
            self.calculate_net.load_state_dict(self.share_model.state_dict())
                
            mini_new_log_probs,mini_entropys,mini_new_values = self.get_prob_entropy_value(mini_states,mini_actions.T,mini_masks)
                        
            mini_advantage = advantage_list[start_index:end_index]
            mini_returns = returns_list[start_index:end_index]
            
            #discrete ratio
            ratio1 = torch.exp(mini_new_log_probs-mini_old_log_probs)
            
            #prod ratio
            #ratio2 = torch.exp(t_new_log_probs.sum(1) - old_log_probs.sum(1)).reshape(-1,1).expand_as(ratio1)
            
            ratio2 = ratio1.prod(1,keepdim=True).expand_as(ratio1)
            #ratio2 = AlgoBase.GradCoef.apply(ratio2,1.0/ratio2.shape[1])
            
            #ratio2 = self.get_prod_ratio(ratio1)
            
            #mixed ratio
            #ratio3 = (AlgoBase.GradCoef.apply(ratio1,ratio_coef) + AlgoBase.GradCoef.apply(ratio2, 2.0 - ratio_coef)) / 2
            ratio3 = ratio1 * ratio_coef + ratio2 * (1.0 - ratio_coef)
            
            #discrete
            if pg_loss_type == 0:
                pg_loss = self.get_pg_loss(ratio1,mini_advantage)
                
            #prod
            elif pg_loss_type == 1:
                pg_loss = self.get_pg_loss(ratio2,mini_advantage)
                
            #mixed
            elif pg_loss_type == 2:
                pg_loss = self.get_pg_loss(ratio3,mini_advantage)
                
            #last_mixed
            elif pg_loss_type == 3:
                pg_loss1 = self.get_pg_loss(ratio1,mini_advantage)
                pg_loss2 = self.get_pg_loss(ratio2,mini_advantage)
                pg_loss = (pg_loss1+pg_loss2)/2
                
            # Policy loss
            pg_loss = -torch.mean(pg_loss)
            
            entropy_loss = -torch.mean(mini_entropys)
            
            v_loss = F.mse_loss(mini_new_values, mini_returns)

            loss = pg_loss + ent_coef * entropy_loss + v_loss*vf_coef

            self.calculate_net.zero_grad()

            #start_time = time.time()
            loss.backward()
            #end_time = time.time()-start_time
            #print('backward_time:',str(end_time))
            
            grads = [
                param.grad.data.cpu().numpy()
                if param.grad is not None else None
                for param in self.calculate_net.parameters()
            ]
                
            # Updating network parameters
            for param, grad in zip(self.share_model.parameters(), grads):
                param.grad = torch.FloatTensor(grad)
                
            if train_config['enable_grad_norm']:
                torch.nn.utils.clip_grad_norm_(self.share_model.parameters(),grad_norm)
            self.share_optim.step()
                                                       
    def get_pg_loss(self,ratio,advantage):
        """Calculate policy gradient loss
        If :math:`enable_clip_max` is false, then ratio between :math:`1 - clip_coef` to :math:`1 + clip_coef`, otherwise is equal to 0
        else ratio between :math:`1- clip_coef` to :math:`min( 1 + clip_coef, max_clip_coef)`, otherwise is equal to 0
        """
        clip_coef = train_config['clip_coef']
        max_clip_coef = train_config['max_clip_coef']
        enable_clip_max = train_config['enable_clip_max']
        
        # base_value = ratio * advantage
        # clip_value = torch.clamp(ratio,1.0 - clip_coef,1.0 + clip_coef) * advantage
        # min_loss_policy = torch.min(base_value, clip_value)        
        # max_loss_policy = torch.max(min_loss_policy,max_clip_coef * advantage)
        
        # return torch.where(advantage>=0,min_loss_policy,max_loss_policy)
        
        positive = torch.where(ratio >= 1.0 + clip_coef, 0 * advantage,advantage)
        if enable_clip_max:
            negtive = torch.where(ratio <= 1.0 - clip_coef,0 * advantage,torch.where(ratio >= max_clip_coef, 0 * advantage,advantage))
        else:
            negtive = torch.where(ratio <= 1.0 - clip_coef,0 * advantage,advantage)
        
        return torch.where(advantage>=0,positive,negtive)*ratio
        
    def get_prob_entropy_value(self,states, actions, masks):
        """
        Calculate log probilities of actions, entropy of distributions and state-values of states 

        Args:
            states
            actions
            masks
        
        Returns:
            log_probs
            entropy
            values
            
        """
        #start_time = time.time()
        distris = self.calculate_net.get_distris(states)
        #end_time = time.time()-start_time
        #print('forward_time:',str(end_time))
        
        values = self.calculate_net.get_value(states)
        action_masks = torch.split(masks, train_config['action_shape'], dim=1)
        distris = [dist.update_masks(mask,device=self.device) for dist,mask in zip(distris,action_masks)]
        log_probs = torch.stack([dist.log_prob(action) for dist,action in zip(distris,actions)])
        entropys = torch.stack([dist.entropy() for dist in distris])
        return log_probs.T, entropys.T, values

    def get_ent_coef(self,version):
        """Decrease entropy coefficient:
        :math:`ef = lr(1- ve / max_ve )`
        where :math:`ef` is entropy coefficient, :math:`ve` is current version and :math:`max_ve` is the highest version.
        Minimum learning rate is equal to 1e-8
        """
        if train_config['enable_entropy_decay']:
            ent_coef = train_config['ent_coef'] * (1 - version*1.0 / train_envs[current_env_name].max_version)
            if ent_coef <= 1e-8:
                ent_coef = 1e-8
            return ent_coef
        else:
            return train_config['ent_coef'] 

    def get_ratio_coef(self,version):
        """increase ratio from 0 to 0.95 in mixed environment"""
        if train_config['enable_ratio_decay']:
            ratio_coef = version/train_envs[current_env_name].max_version
            if ratio_coef >= 1.0:
                ratio_coef = 0.95       
            return ratio_coef   
        
        else:
            return train_config['ratio_coef']
        
if __name__ == "__main__":
    # used for tensorboard naming
    comment = "_PPOMicroRTSShare_" + current_env_name + "_" + train_config['tensorboard_comment']
    writer = SummaryWriter(comment=comment)
    
    # initialize training network
    train_net = PPOMicroRTSShareNet()

    # set model dictionary
    model_dict = {}
    model_dict[0] = 0
    model_dict['num_trainer'] = 1
    model_dict['train_version'] = 0

    # initialize a RL agent, smaple agent used for sampling training data, check agent used for evaluating 
    # and calculate used for calculating gradients
    sample_agent = PPOMicroRTSShareAgent(train_net,model_dict,is_checker=False)
    check_agent = PPOMicroRTSShareAgent(train_net,model_dict,is_checker=True)
    calculate = PPOMicroRTSShareCalculate(train_net,model_dict,0)

    # dimention of a state and an action
    state = torch.randn(1,10,10,27)
    action = torch.randn(1,8)

    # hyperparameters
    MAX_VERSION = 1500
    REPEAT_TIMES = 10
    for _ in range(MAX_VERSION):
        # Sampling training data and calculating time cost
        start_time = time.time()
        samples_list = sample_agent.sample_env()
        end_time = time.time()-start_time
        print('sample_time:',end_time)
        samples = []
        
        for s in samples_list:
            samples.append(s)
        
        # Calculating policy gradients and time cost
        start_time = time.time()
        calculate.begin_batch_train(samples)
        for _ in range(REPEAT_TIMES):
            calculate.generate_grads()
        calculate.end_batch_train()
        end_time = time.time()-start_time                    
        print('calculate_time:',end_time)
        
        # Updating model version
        model_dict[0] = model_dict[0] + 1
        model_dict['train_version'] = model_dict[0]
        
        # Evaluating agent
        infos = check_agent.check_env()
        for (key,value) in  infos.items():
            writer.add_scalar(key, value, model_dict[0])
            
        print("version:",model_dict[0],"sum_rewards:",infos['sum_rewards'])