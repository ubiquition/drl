import sys,os
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
    '10_10':SimpleNamespace(**{'map_size': 10*10,'conv_linear':3*3,'map_path':'maps/10x10/basesWorkers10x10.xml','num_steps':512,'max_version':int(1e6)}),
    '16_16':SimpleNamespace(**{'map_size': 16*16,'conv_linear':6*6,'map_path':'maps/16x16/basesWorkers16x16.xml','num_steps':512,'max_version':int(1e6)}),
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

# policy gradient type: 0 is discrete, 1 is continues, 2 is mixed and 3 is mixed policy gradient loss
train_config['pg_loss_type'] = 0 

# whether clip value loss
train_config['enable_clip_max'] = True 

# whether to decay ratio 
train_config['enable_ratio_decay'] = False 

# whether to decay entropy coefficient
train_config['enable_entropy_decay'] = False 

# whether to decay learning rate
train_config['enable_lr_decay'] = False

# whether to normalize advantage function
train_config['enable_adv_norm'] = False 

# the number of environments
train_config['num_envs'] = 1 

# action space of environments
train_config['action_shape'] = [train_envs[current_env_name].map_size, 6, 4, 4, 4, 4, 7, 49] 

# for tensorboard naming
train_config['tensorboard_comment'] = 'other_infos'

class PPOMicroRTSHogwildNet(AlgoBase.AlgoBaseNet):
    """Policy class used with PPO

    Example:
    >>> current_env_name = '10_10' # environment name
    >>> train_net = PPOMicroRTSHogwildNet()
    >>> state = torch.randn(1,10,10,27) # one state dimention
    >>> action = torch.randn(1,8) # one action dimention
    >>> train_net.get_distris(state)
    >>> distris, value = train_net(state)
    """
    def __init__(self):
        super(PPOMicroRTSHogwildNet,self).__init__()

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
            >>> train_net = PPOMicroRTSHogwildNet()
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

        Returns:
            probability distributions and state values 

        Example:
            >>> current_env_name = '10_10' # environment name
            >>> train_net = PPOMicroRTSHogwildNet()
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
            >>> train_net = PPOMicroRTSHogwildNet()
            >>> state = torch.randn(1,10,10,27) # one state dimention
            >>> train_net.get_value(state)
        """
        # Moving last convolution channel shape to the second dimention 
        states = states.permute((0, 3, 1, 2))
        return self.value(states)
    
class PPOMicroRTSHogwildUtils(AlgoBase.AlgoBaseUtils):
    pass
        
class PPOMicroRTSHogwildAgent(AlgoBase.AlgoBaseAgent):
    """
    Agent class used with PPO, allowing collect data and evaluate agents.

    Args:
        sample_net: policy network (default: PPOMicroRTSHogwildNet)
        model_dict: a dict of model configuration
        is_checker: if "True", then evaluating the agent through running 1024 timesteps with 
            the highest probility of action, else collecting the training data.

    Example:
        >>> train_net = PPOMicroRTSHogwildNet()
        >>> # Collecting training data
        >>> sample_agent = PPOMicroRTSHogwildAgent(train_net,model_dict,is_checker=False)
        >>> transition = sample_agent.sample_env()
        >>> # Evaluating agent 
        >>> check_agent = PPOMujocoNormalHogwildAgent(train_net,model_dict,is_checker=True)
        >>> info = check_agent.check_env()
        >>> print(info['sum_rewards'], info['mean_entropys'], info['mean_log_probs'], ['mean_win_rates'])

    """
    def __init__(self,sample_net:PPOMicroRTSHogwildNet,model_dict,is_checker=False):
        super(PPOMicroRTSHogwildAgent,self).__init__()
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
            print("PPOMicroRTSHogwild check map is",current_env_name)
        self.obs = self.env.reset()
        
    def get_comment_info(self):
        return current_env_name + "_" + train_config['tensorboard_comment']
        
    def sample_env(self):
        """
        Collect training data

        Example:
            >>> train_net = PPOMicroRTSHogwildNet()
            >>> sample_agent = PPOMicroRTSHogwildAgent(train_net,model_dict,is_checker=False)
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
            >>> train_net = PPOMicroRTSHogwildNet()
            >>> check_agent = PPOMicroRTSHogwildAgent(train_net,model_dict,is_checker=True)
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
            >>> train_net = PPOMicroRTSHogwildNet()
            >>> sample_agent = PPOMicroRTSHogwildAgent(train_net,model_dict,is_checker=False)
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
            >>> train_net = PPOMicroRTSHogwildNet()
            >>> check_agent = PPOMicroRTSHogwildAgent(train_net,model_dict,is_checker=True)
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
    
class PPOMicroRTSHogwildCalculate(AlgoBase.AlgoBaseCalculate):
    """
    Training calss used with PPO

    Args:
        share_model: policy network (default: PPOMicroRTSHogwildNet)
        model_dict: a dict of model configuration
        calculate_index: the :math:`calculate_index`th agent for training

    Example:
        >>> train_net = PPOMicroRTSHogwildNet()
        >>> # Collecting training data
        >>> calculate = PPOMicroRTSHogwildCalculate(train_net, model_dict, calculate_index)
        >>> # samples are from transitions
        >>> calculate.begin_batch_train(samples)
        >>> for _ in range(REPEAT_TIMES):
        >>>    calculate.generate_grads()
        >>> calculate.end_batch_train()
    """
    def __init__(self,share_model:PPOMicroRTSHogwildNet,model_dict,calculate_index):
        super(PPOMicroRTSHogwildCalculate,self).__init__()
        self.device = torch.device('cpu')
        self.model_dict = model_dict
        self.calculate_number = self.model_dict['num_trainer']
        self.calculate_index = calculate_index
        
        self.calculate_net = share_model
        self.calculate_net.to(self.device)
    
        self.calculate_optim = torch.optim.Adam(params=self.calculate_net.parameters(), lr=train_config['learning_rate'])
        
        self.samples = None
        
    def begin_batch_train(self,samples_list:list):
        """
        Store training data

        Example:
            >>> train_net = PPOMicroRTSHogwildNet()
            >>> calculate = PPOMicroRTSHogwildCalculate(train_net, model_dict, calculate_index)
            >>> calculate.begin_batch_train(samples)
        """
        samples = []
        for samples_item in samples_list:
            samples.extend(samples_item)
            
        self.samples = samples
        
    def end_batch_train(self):
        """
        Clear training data and update learning rate

        Example:
            >>> train_net = PPOMicroRTSHogwildNet()
            >>> calculate = PPOMicroRTSHogwildCalculate(train_net, model_dict, calculate_index)
            >>> calculate.end_batch_train(samples)
        """
        self.samples = None
        train_version = self.model_dict[self.calculate_index]
        self.decay_lr(train_version)
        
    def decay_lr(self, version):
        """
        Decrease learning rate:
        :math:`lr = lr(1- ve / max_ve )`
        where :math:`lr` is learning rate, :math:`ve` is current version and :math:`max_ve` is the highest version.
        Minimum learning rate is equal to 1e-6

        Example:
            >>> train_net = PPOMicroRTSHogwildNet()
            >>> calculate = PPOMicroRTSHogwildCalculate(train_net, model_dict, calculate_index)
            >>> calculate.decay_lr(calculate_index)

        """
        if train_config['enable_lr_decay']:
            lr_now = train_config['learning_rate'] * (1 - version*1.0 / train_envs[current_env_name].max_version)
            if lr_now <= 1e-6:
                lr_now = 1e-6
            
            if self.calculate_optim is not None:
                for param in self.calculate_optim.param_groups:
                    param['lr'] = lr_now
        
    def generate_grads(self):    
        """
        Update share network parameters

        Example:
            >>> train_net = PPOMicroRTSHogwildNet()
            >>> calculate = PPOMicroRTSHogwildCalculate(train_net, model_dict, calculate_index)
            >>> calculate.begin_batch_train(samples)
            >>> REPEAT_TIMES = 10
            >>> for_ in range(REPEAT_TIMES = 10):
            >>>     calculate.generate_grads()
            >>> calculate.end_batch_train()
        """
        self.calculate_optim.zero_grad()
        self.generate_samples_grads(self.samples)
        self.calculate_optim.step()
                                                             
    def generate_samples_grads(self,samples):
        """Calculate policy gradient loss and backward propagation, where :math:`Loss = L^{pg} + c_1 * L^{VF} + C_2 * S`,
        :math: `L^{pg}` is policy gradient loss, :math: `L^{VF}` is value loss, :math: `S` is entropy loss and :math:`c_1, c_2`
        are coefficent, respectively. 
        """         
        train_version = self.model_dict['train_version']
        gamma = train_config['gamma']
        gae_lambda = train_config['gae_lambda']
        pg_loss_type = train_config['pg_loss_type']
                            
        ent_coef = self.get_ent_coef(train_version)
        vf_coef = train_config['vf_coef']
        ratio_coef = self.get_ratio_coef(train_version)
                
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s[4] for s in samples])
        
        s_states = torch.Tensor(np.array([s[0] for s in samples])).to(self.device)
        s_actions = torch.Tensor(np.array([s[1] for s in samples])).to(self.device)
        s_masks = torch.Tensor(np.array([s[3] for s in samples])).to(self.device)
        s_log_probs = torch.Tensor(np.array([s[5] for s in samples])).to(self.device)
        
        b_new_log_prob, b_entropy, b_new_values = self.get_prob_entropy_value(s_states, actions=s_actions.T, masks=s_masks)
        
        #start = timer()
        np_advantages,np_returns = AlgoBase.calculate_gae(b_new_values.detach().cpu().numpy().reshape(-1),s_rewards,s_dones,gamma,gae_lambda)
        #run_time = timer() - start
        #print("CPU function took %f seconds." % run_time)
        
        if train_config['enable_adv_norm']:
            np_advantages = (np_advantages - np_advantages.mean()) / (np_advantages.std() + 1e-8)
                                                                                    
        b_advantages = torch.Tensor(np_advantages).reshape(-1,1).to(self.device)
        b_returns = torch.Tensor(np_returns).reshape(-1,1).to(self.device)
                                                
        #discrete ratio
        ratio1 = torch.exp(b_new_log_prob-s_log_probs)
        
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
             pg_loss = self.get_pg_loss(ratio1,b_advantages)
             
        #prod
        elif pg_loss_type == 1:
             pg_loss = self.get_pg_loss(ratio2,b_advantages)
             
        #mixed
        elif pg_loss_type == 2:
            pg_loss = self.get_pg_loss(ratio3,b_advantages)
            
        #last_mixed
        elif pg_loss_type == 3:
            pg_loss1 = self.get_pg_loss(ratio1,b_advantages)
            pg_loss2 = self.get_pg_loss(ratio2,b_advantages)
            pg_loss = (pg_loss1+pg_loss2)/2
                        
        # Policy loss
        pg_loss = -torch.mean(pg_loss)
        
        entropy_loss = -torch.mean(b_entropy)
        
        v_loss = F.mse_loss(b_new_values, b_returns)

        loss = pg_loss + ent_coef * entropy_loss + v_loss*vf_coef

        loss.backward()
                       
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
        Calculate log probilities of action, entropy of distribution and state values of each state

        Args:
            states
            actions
            masks
        
        Returns:
            log_probs
            entropy
            values

        """
        distris,values = self.calculate_net(states)
        action_masks = torch.split(masks, train_config['action_shape'], dim=1)
        distris = [dist.update_masks(mask) for dist,mask in zip(distris,action_masks)]
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

    # initialize training network
    net = PPOMicroRTSHogwildNet()

    # set model dictionary
    model_dict = {}

    # initialize a RL agent, agent used for sampling training data, check agent used for evaluating 
    # and calculate used for calculating gradients
    agent = PPOMicroRTSHogwildAgent(net,model_dict,False)
    check_agent = PPOMicroRTSHogwildAgent(net,model_dict,True)
    calculate = PPOMicroRTSHogwildCalculate(net,model_dict)