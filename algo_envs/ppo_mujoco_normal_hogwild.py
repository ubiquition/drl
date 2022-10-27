import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch 
import torch.nn as nn
import gym
import numpy as np
from gym.spaces.box import Box
from torch.distributions.normal import Normal
from torch.nn import functional as F
from types import SimpleNamespace
import algo_envs.algo_base as AlgoBase
import torch.distributed as dist

"""
Environments' objects for training, including
     env_name:  environment name
      obs_dim:  dimention of observation
      act_dim:  dimention of action
     hide_dim:  dimention of hidden network
     ent_coef:  entropy coefficient
  max_version:  maximum number of episodes
    use_noise:  whether to use noise network

Example:
    >>> obs_dim = train_envs[current_env_name].obs_dim
    >>> act_dim = train_envs[current_env_name].act_dim
    >>> hide_dim = train_envs[current_env_name].hide_dim
"""

train_envs = {
    'Swimmer':SimpleNamespace(**{'env_name': "Swimmer-v3",'obs_dim':8,'act_dim':2,'hide_dim':64,'ent_coef':0.01,'max_version':int(1e6),'use_noise':True}),
    'HalfCheetah':SimpleNamespace(**{'env_name': "HalfCheetah-v3",'obs_dim':17,'act_dim':6,'hide_dim':64,'ent_coef':0.01,'max_version':int(1e6),'use_noise':True}),
    'Ant':SimpleNamespace(**{'env_name': "Ant-v3",'obs_dim':111,'act_dim':8,'hide_dim':256,'ent_coef':0.01,'max_version':int(1e6),'use_noise':True}),
    'Hopper':SimpleNamespace(**{'env_name': "Hopper-v3",'obs_dim':11,'act_dim':3,'hide_dim':64,'ent_coef':0.01,'max_version':int(1e6),'use_noise':True}),
    'Pusher':SimpleNamespace(**{'env_name': "Pusher-v2",'obs_dim':23,'act_dim':7,'hide_dim':64,'ent_coef':0.01,'max_version':int(1e6),'use_noise':True}),
    'Humanoid':SimpleNamespace(**{'env_name': "Humanoid-v3",'obs_dim':376,'act_dim':17,'hide_dim':512,'ent_coef':0.01,'max_version':int(5e5),'use_noise':True}),
    'Walker2d':SimpleNamespace(**{'env_name': "Walker2d-v3",'obs_dim':17,'act_dim':6,'hide_dim':64,'ent_coef':0.01,'max_version':int(1e6),'use_noise':True}),
}

# current environment name
current_env_name = 'Swimmer'

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

#train_config['ent_coef'] = 0.01# weight of entropy

# weight of value loss
train_config['vf_coef'] = 4 

# learning rate
train_config['learning_rate'] = 2.5e-4 

# weight of ratio
train_config['ratio_coef'] = 0.5 

# policy gradient type: 0 is discrete, 1 is continues, 2 is mixed and 3 is mixed policy gradient loss
train_config['pg_loss_type'] = 2 

# whether clip value loss
train_config['enable_clip_max'] = True 

# whether to decay ratio when use mixed environment
train_config['enable_ratio_decay'] = False 

# whether to decay entropy coefficient
train_config['enable_entropy_decay'] = False 

# whether to decay learning rate
train_config['enable_lr_decay'] = False 

# whether to normalize advantage function
train_config['enable_adv_norm'] = False 

# the num of environments
train_config['num_envs'] = 1 

# an episode length
train_config['num_steps'] = 512 

# for tensorboard naming
train_config['tensorboard_comment'] = 'other_infos'

class PPOMujocoNormalHogwildNet(AlgoBase.AlgoBaseNet):
    """ Policy class used with continues PPO


    Example:
        >>> current_env_name = 'Ant' # Mujoco environment
        >>> train_net = PPOMujocoNormalHogwildNet()
        >>> states = torch.randn(64,111) # minibatch = 64, state dimention = 111
        >>> actions = torch.rand(64,8) # action dimention = 8 
        >>> train_net.get_distris(states) # get policy distributions
        Normal(loc: torch.Size([64,8]), scale: torch.Size([64,8]))
        
        >>> values, log_probs, distris_entropy = train_net(states,actions) # return state values, log probilities of actions and distribution entropy of actions
        >>> print(values.size(), log_probs.size(), distris_entropy.size())
        torch.Size([64,1]) torch.Size([64,8]) torch.Size([64,8])
    
    """
    def __init__(self):
        super(PPOMujocoNormalHogwildNet,self).__init__()
        
        obs_dim = train_envs[current_env_name].obs_dim
        act_dim = train_envs[current_env_name].act_dim
        hide_dim = train_envs[current_env_name].hide_dim
        
        if train_envs[current_env_name].use_noise:
            self.noise_layer_out = AlgoBase.NoisyLinear(hide_dim,act_dim)
            self.noise_layer_hide = AlgoBase.NoisyLinear(hide_dim,hide_dim)
                            
            #normal mu
            self.mu = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    self.noise_layer_hide,
                    nn.ReLU(),
                    self.noise_layer_out,
                    nn.Tanh()
                )
        else:
            #normal mu
            self.mu = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, act_dim)),
                    nn.Tanh()
                )
                
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        
        self.value = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, 1))
            )
                
    def get_distris(self,states):
        """
        Calculate the distributions of states
        
        Args:
            states

        Return:
            distribution of states

        Example:
            >>> current_env_name = 'Ant' 
            >>> train_net = PPOMujocoNormalHogwildNet()
            >>> states = torch.randn(64,111) 
            >>> train_net.get_distris(states) 
            Normal(loc: torch.Size([64,8]), scale: torch.Size([64,8]))
        """
        mus = self.mu(states)
        distris = Normal(mus,torch.exp(self.log_std))
        return distris
        
    def forward(self,states,actions):
        """
        Calculate state values, log probilities of each action and distribution entropy of each action

        Args:
            states
            actions

        Return:
            state-values, log probilities of each action and distribution entropy of each action

        Example:
            >>> current_env_name = 'Ant' # Mujoco environment
            >>> train_net = PPOMujocoNormalHogwildNet()
            >>> states = torch.randn(64,111) 
            >>> actions = torch.rand(64,8) 
            >>> values, log_probs, distris_entropy = train_net(states,actions) 
            >>> print(values.size(), log_probs.size(), distris_entropy.size())
            torch.Size([64,1]) torch.Size([64,8]) torch.Size([64,8])
        """
        # mus = self.mu(states)
        # return mus
        values = self.value(states)
        distris = self.get_distris(states)
        log_probs = distris.log_prob(actions) 
        return values,log_probs,distris.entropy()
    
    def get_sample_data(self,states):
        """
        Return actions and log probilities of each action

        Args:
            states

        Example:
            >>> current_env_name = 'Ant' 
            >>> train_net = PPOMujocoNormalHogwildNet()
            >>> states = torch.randn(64,111)
            >>> actions, log_probs = train_net.get_sample_data(states) 
        """
        distris = self.get_distris(states)
        actions = distris.sample()
        log_probs = distris.log_prob(actions)
        return actions,log_probs
    
    def get_check_data(self,states):
        """
        Return expectations of states, entropy of state distributions and log probilities of taking the best actions

        Args:
            states

        Example:
            >>> current_env_name = 'Ant' 
            >>> train_net = PPOMujocoNormalHogwildNet()
            >>> states = torch.randn(64,111)
            >>> mus, entropy, log_probs = train_net.get_check_data(states) 
        """
        distris = self.get_distris(states)
        mus = self.mu(states)
        log_probs = distris.log_prob(distris.mean)
        return mus,distris.entropy(),log_probs
    
    def get_calculate_data(self,states,actions):
        values = self.value(states)
        distris = self.get_distris(states)
        log_probs = distris.log_prob(actions) 
        return values,log_probs,distris.entropy()
    
    def sample_noise(self):
        """Add normal noise to network parameter, more details see in NoisyLinear class 

        Example:
            >>> current_env_name = 'Ant' 
            >>> train_net = PPOMujocoNormalHogwildNet()
            >>> train_net.sample_noise()
        """
        if train_envs[current_env_name].use_noise:
            self.noise_layer_out.sample_noise()
            self.noise_layer_hide.sample_noise()
    
class PPOMujocoNormalHogwildUtils(AlgoBase.AlgoBaseUtils):
    pass
                    
class PPOMujocoNormalHogwildAgent(AlgoBase.AlgoBaseAgent):
    """
    Agent class used with continues PPO, allowing collect data and evaluate agents.

    Args:
        sample_net: policy network (default: PPOMujocoNormalHogwildNet)
        model_dict: a dict of model configuration
        is_checker: if "True", then evaluating the agent through running 1024 timesteps with 
        the highest probility of action, else collecting the training data.

    Example:
        >>> train_net = PPOMujocoNormalHogwildNet()
        >>> # Collecting training data
        >>> sample_agent = PPOMujocoNormalHogwildAgent(train_net,model_dict,is_checker=False)
        >>> transition = sample_agent.sample_env()
        >>> # Evaluating agent 
        >>> check_agent = PPOMujocoNormalShareAgent(train_net,model_dict,is_checker=True)
        >>> info = check_agent.check_env()
        >>> print(info['sum_rewards'], info['mean_entropys'], info['mean_mus'], info['mean_log_probs'])

    """
    
    def __init__(self,sample_net:PPOMujocoNormalHogwildNet,model_dict,is_checker):
        super(PPOMujocoNormalHogwildAgent,self).__init__()
        self.sample_net = sample_net
        self.model_dict = model_dict
        self.num_steps = train_config['num_steps']
        self.num_envs = train_config['num_envs']
        self.rewards = []
        
        env_name = train_envs[current_env_name].env_name
    
        if not is_checker:
            self.envs = [gym.make(env_name) for _ in range(self.num_envs)]
            self.states = [self.envs[i].reset() for i in range(self.num_envs)]
        else:
            print("PPOMujocoNormalHogwild check mujoco env is",env_name)
            self.envs = gym.make(env_name)
            self.states = self.envs.reset()
            self.num_steps = 1024
            
    def get_comment_info(self):
        return current_env_name + "_" + train_config['tensorboard_comment']
        
    def sample_env(self):
        """collect training data 
        Example:
            >>> train_net = PPOMujocoNormalHogwildNet()
            >>> sample_agent = PPOMujocoNormalHogwildAgent(train_net,model_dict,is_checker=False)
            >>> transition = sample_agent.sample_env()
        """
    
        exps=[[] for _ in range(self.num_envs)]

        for step in range(self.num_steps):
            
            actions,log_probs = self.get_sample_actions(self.states)
            for i in range(self.num_envs):
                next_state_n, reward_n, done_n, _ = self.envs[i].step(actions[i])                
                if done_n:
                    next_state_n = self.envs[i].reset()
                    
                if done_n or step == self.num_steps-1:
                    done = True
                else:
                    done = False
                    
                exps[i].append([self.states[i],actions[i],reward_n,done,log_probs[i],self.model_dict['train_version']])
                self.states[i] = next_state_n
                
        return exps
    
    def check_env(self):
        """evaluate agent
        Example:
            >>> train_net = PPOMujocoNormalHogwildNet()
            >>> check_agent = PPOMujocoNormalHogwildAgent(train_net,model_dict,is_checker=True)
            >>> info = check_agent.check_env()
        """
        step_record_dict = dict()
        
        is_done = False
        steps = 0
        mus = []
        rewards = []
        entropys = []
        log_probs = []

        while True:
            #self.envs.render()
            mu,entropy,log_prob = self.get_check_action(self.states)
            next_state_n, reward_n, is_done, _ = self.envs.step(mu)
            if is_done:
                next_state_n = self.envs.reset()
            self.states = next_state_n
            rewards.append(reward_n)
            mus.append(mu)
            entropys.append(entropy)
            log_probs.append(log_prob)
            
            steps += 1
            if is_done:
                break
            #if steps >= self.num_steps:
            #    break
        
        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['mean_entropys'] = np.mean(entropys)
        step_record_dict['mean_mus'] = np.mean(mus)
        step_record_dict['mean_log_probs'] = np.mean(log_probs)
        
        return step_record_dict
            
    @torch.no_grad()
    def get_sample_actions(self,states):
        """Sample actions and calculate action probilities of action

        Args:
            states

        Returns:
            actions
            log_probs

        Example:
            >>> train_net = PPOMujocoNormalHogwildNet()
            >>> sample_agent = PPOMujocoNormalHogwildAgent(train_net,model_dict,is_checker=False)
            >>> states = torch.randn(64,111) 
            >>> actions, log_probs = sample_agent.get_sample_actions(states)
        """
        states_v = torch.Tensor(np.array(states))
        actions,log_probs = self.sample_net.get_sample_data(states_v)
        return actions.cpu().numpy(),log_probs.cpu().numpy()
    
    @torch.no_grad()
    def get_check_action(self,state):
        """Get the highest probility of action, and it's entropy and log probility 
        Example:
            >>> train_net = PPOMujocoNormalHogwildNet()
            >>> check_agent = PPOMujocoNormalHogwildAgent(train_net,model_dict,is_checker=True)
            >>> states = torch.randn(111) 
            >>> mu, entropy, log_prob = check_agent.get_check_actions(state)
        """
        state_v = torch.Tensor(np.array(state))
        mu,entropy,log_prob = self.sample_net.get_check_data(state_v)
        return mu.cpu().numpy(),entropy.cpu().numpy(),log_prob.cpu().numpy()
            
class PPOMujocoNormalHogwildCalculate(AlgoBase.AlgoBaseCalculate):
    """
    training class used with continues PPO

    Args:
        share_model: policy network (default: PPOMujocoNormalHogwildNet)
        model_dict: a dict of model configuration
        calculate_index: the :math:`calculate_index`th agent for training

    
    Example:
        >>> train_net = PPOMujocoNormalHogwildNet()
        >>> calculate = PPOMujocoNormalHogwildCalculate(train_net,model_dict,calculate_index)
        >>> # samples are from transitions
        >>> calculate.begin_batch_train(samples)
        >>> for _ in range(REPEAT_TIMES):
        >>>    calculate.generate_grads()
        >>> calculate.end_batch_train()

    """
    
    def __init__(self,share_model:PPOMujocoNormalHogwildNet,model_dict,calculate_index):
        super(PPOMujocoNormalHogwildCalculate,self).__init__()
        self.model_dict = model_dict
        self.device = torch.device('cpu')
        self.calculate_number = self.model_dict['num_trainer']
        self.calculate_index = calculate_index
                        
        self.calculate_net = share_model
        self.calculate_net.to(self.device)
    
        self.calculate_optim = torch.optim.Adam(params=self.calculate_net.parameters(), lr=train_config['learning_rate'])
        
    def begin_batch_train(self,samples_list:list):
        """store training data
        Example:
            >>> train_net = PPOMujocoNormalHogwildNet()
            >>> calculate = PPOMujocoNormalHogwildCalculate(train_net,model_dict,calculate_index)
            >>> calculate.begin_batch_train(samples)
        """
        samples = []
        for samples_item in samples_list:
            samples.extend(samples_item)
            
        self.samples = samples
        
    def end_batch_train(self):
        """clear training data, update learning rate and noisy network
        Example:
            >>> train_net = PPOMujocoNormalHogwildNet()
            >>> calculate = PPOMujocoNormalHogwildCalculate(train_net,model_dict,calculate_index)
            >>> calculate.end_batch_train(samples)
        """
        self.samples = None
        train_version = self.model_dict[self.calculate_index]
        self.decay_lr(train_version)
        
        # Resetting sample noise
        if self.calculate_index == self.calculate_number - 1:
            self.calculate_net.sample_noise()
            
    def decay_lr(self, version):
        """decrease learning rate:
        :math:`lr = lr(1- ve / max_ve )`
        where :math:`lr` is learning rate, :math:`ve` is current version and :math:`max_ve` is the highest version.
        Minimum learning rate is equal to 1e-6

        Example:
            >>> train_net = PPOMujocoNormalHogwildNet()
            >>> calculate = PPOMujocoNormalHogwildCalculate(train_net,model_dict,calculate_index)
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
        """ update share network parameters. 

        Example:
            >>> train_net = PPOMujocoNormalHogwildNet()
            >>> calculate = PPOMujocoNormalHogwildCalculate(train_net,model_dict,calculate_index)
            >>> calculate.begin_batch_train(samples)
            >>> REPEAT_TIMES = 10
            >>> for _ in range(REPEAT_TIMES):
            >>>     calculate.generate_grads()
            >>> calculate.end_batch_train()
        """  
        self.calculate_optim.zero_grad()
        self.generate_samples_grads(self.samples)
        self.calculate_optim.step()
                                                                             
    def generate_samples_grads(self,samples):
        """
        If action is discrete, then :math:`ratio1 = exp(new_log_probs - old_log_probs)`, if action is continues, then
        :math:`ratio2 = \prod{ratio1}` and expand to the same dimention as :math:`ratio1`, if action is mixed, then 
        :math:`ratio3 = ratio1 * ratio_coef + ratio2 * (1.0 - ratio_coef)`, where :math:`ratio_coef` is weight coefficent.
        
        """
        train_version = self.model_dict[self.calculate_index]
        gamma = train_config['gamma']
        gae_lambda = train_config['gae_lambda']
        vf_coef = train_config['vf_coef']
        pg_loss_type = train_config['pg_loss_type']

        ent_coef = train_envs[current_env_name].ent_coef
        ratio_coef = self.get_ratio_coef(train_version)
    
        s_states = np.array([s[0] for s in samples])
        s_actions = np.array([s[1] for s in samples])
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s[3] for s in samples])
        s_log_probs = np.array([s[4] for s in samples])
        #s_versions = [s[5] for s in samples]

        t_states = torch.Tensor(s_states).to(self.device)
        t_actions = torch.Tensor(s_actions).to(self.device)
        old_log_probs = torch.Tensor(s_log_probs).to(self.device)
        
        t_new_values,t_new_log_probs,t_entropys = self.calculate_net(t_states,t_actions)
        
        #start = timer()
        np_advantages,np_returns = AlgoBase.calculate_gae(t_new_values.detach().cpu().numpy().reshape(-1),s_rewards,s_dones,gamma,gae_lambda)
        #run_time = timer() - start
        #print("CPU function took %f seconds." % run_time)
        
        if train_config['enable_adv_norm']:
            np_advantages = (np_advantages - np_advantages.mean()) / (np_advantages.std() + 1e-8)
                                                    
        t_advantages = torch.Tensor(np_advantages).to(self.device)        
        t_returns = torch.Tensor(np_returns).to(self.device)
                
        t_advantages = t_advantages.reshape(-1,1)
        t_returns = t_returns.reshape(-1,1)
                
        t_new_log_probs = t_new_log_probs.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        #discrete ratio
        ratio1 = torch.exp(t_new_log_probs-old_log_probs)

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
             pg_loss = self.get_pg_loss(ratio1,t_advantages)
             
        #prod
        elif pg_loss_type == 1:
             pg_loss = self.get_pg_loss(ratio2,t_advantages)
             
        #mixed
        elif pg_loss_type == 2:
            pg_loss = self.get_pg_loss(ratio3,t_advantages)
            
        #last_mixed
        elif pg_loss_type == 3:
            pg_loss1 = self.get_pg_loss(ratio1,t_advantages)
            pg_loss2 = self.get_pg_loss(ratio2,t_advantages)
            pg_loss = (pg_loss1+pg_loss2)/2
                        
        # Policy loss
        pg_loss = -torch.mean(pg_loss)
        
        v_loss = F.mse_loss(t_returns, t_new_values) * vf_coef
        
        e_loss = -torch.mean(t_entropys) * ent_coef
        
        loss = pg_loss + v_loss + e_loss

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
    
    def get_ent_coef(self,version):
        """decrease entropy coefficient:
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
            return train_envs[current_env_name].ent_coef

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
    net = PPOMujocoNormalHogwildNet()

    # set model dictionary
    model_dict = {}

    # initialize a RL agent, smaple agent used for sampling training data, check agent used for evaluating 
    # and calculate used for calculating gradients
    agent = PPOMujocoNormalHogwildAgent(net,model_dict,is_checker=False)
    check_agent = PPOMujocoNormalHogwildAgent(net,model_dict,is_checker=True)
    calculate = PPOMujocoNormalHogwildCalculate(net,model_dict)