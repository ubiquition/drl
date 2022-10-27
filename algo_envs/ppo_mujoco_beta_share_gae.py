import sys,os,time
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch 
import torch.nn as nn
import gym
import numpy as np
from gym.spaces.box import Box
from torch.distributions.normal import Normal
from torch.nn import functional as F
from torch.distributions.beta import Beta
import algo_envs.algo_base as AlgoBase
from types import SimpleNamespace
from timeit import default_timer as timer
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

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

# weight of value loss
train_config['vf_coef'] = 4 

# learning rate
train_config['learning_rate'] = 2.5e-4 

# weight of ratio
train_config['ratio_coef'] = 0.5 

# max gradient normal of value
train_config['grad_norm'] = 0.5 

# policy gradient type: 0 is discrete, 1 is continues, 2 is mixed and 3 is mixed policy gradient loss
train_config['pg_loss_type'] = 2 

# whether clip value loss
train_config['enable_clip_max'] = True 

# whether to decay ratio 
train_config['enable_ratio_decay'] = False 

# whether to decay entropy coefficient
train_config['enable_entropy_decay'] = False 

# whether to decay learning rate
train_config['enable_lr_decay'] = False 

# whether to clip gradient norm
train_config['enable_grad_norm'] = False 

# whether to normalize advantage function
train_config['enable_adv_norm'] = True 

# whether to use mini batch
train_config['enable_mini_batch'] = False 

# the number of environments
train_config['num_envs'] = 1 

# an episode length
train_config['num_steps'] = 512 

# value of max action
train_config['max_action'] = 1.0

# whether to use GPU training
train_config['use_gpu'] = True 

# minibatch size, which should divisible by an epdsode length
train_config['mini_batch_size'] = 128 

# the length of generalized advantage estimator for calculating 
train_config['gae_length'] = 256 

# for tensorboard naming
train_config['tensorboard_comment'] = 'gae_length_256'

class PPOMujocoBetaShareGAENet(AlgoBase.AlgoBaseNet):    
    """ Policy class used with continues PPO


    Example:
        >>> current_env_name = 'Ant' # Mujoco environment
        >>> train_net = PPOMujocoBetaShareGAENet()
        >>> states = torch.randn(64,111) # minibatch = 64, state dimention = 111
        >>> actions = torch.rand(64,8) # action dimention = 8 
        >>> train_net.get_distris(states) # get policy distributions
        Normal(loc: torch.Size([64,8]), scale: torch.Size([64,8]))
        
        >>> values, log_probs, distris_entropy = train_net(states,actions) # return state values, log probilities of actions and distribution entropy of actions
        >>> print(values.size(), log_probs.size(), distris_entropy.size())
        torch.Size([64,1]) torch.Size([64,8]) torch.Size([64,8])
    
    """
    def __init__(self):
        super(PPOMujocoBetaShareGAENet,self).__init__()
        
        obs_dim = train_envs[current_env_name].obs_dim
        act_dim = train_envs[current_env_name].act_dim
        hide_dim = train_envs[current_env_name].hide_dim
                
        if train_envs[current_env_name].use_noise:
            self.alpha_noisy_layers = [AlgoBase.NoisyLinear(hide_dim, act_dim),AlgoBase.NoisyLinear(hide_dim, hide_dim)]
            self.beta_noisy_layers = [AlgoBase.NoisyLinear(hide_dim, act_dim),AlgoBase.NoisyLinear(hide_dim, hide_dim)]
                                        
            self.alpha = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    self.alpha_noisy_layers[1],
                    nn.ReLU(),
                    self.alpha_noisy_layers[0],
                    nn.Softplus()
                )
            
            self.beta = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    self.beta_noisy_layers[1],
                    nn.ReLU(),
                    self.beta_noisy_layers[0],
                    nn.Softplus()
                )
        else:                            
            self.alpha = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, act_dim)),
                    nn.Softplus()
                )
            
            self.beta = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, act_dim)),
                    nn.Softplus()
                )
                
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
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> states = torch.randn(64,111) 
            >>> train_net.get_distris(states) 
            Normal(loc: torch.Size([64,8]), scale: torch.Size([64,8]))
        """
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alphas = self.alpha(states) + 1.0
        betas = self.beta(states) + 1.0
        distris = Beta(alphas,betas)
        return distris
        
    def forward(self,states,actions):
        """
        Calculate state values, log probilities of each action and distribution entropy of each action

        Args:
            states
            actions

        Return:
            state values, log probilities of each action and distribution entropy of each action

        Example:
            >>> current_env_name = 'Ant' # Mujoco environment
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> states = torch.randn(64,111) 
            >>> actions = torch.rand(64,8) 
            >>> values, log_probs, distris_entropy = train_net(states,actions) 
            >>> print(values.size(), log_probs.size(), distris_entropy.size())
            torch.Size([64,1]) torch.Size([64,8]) torch.Size([64,8])
        """
        # distris = self.get_distris(states)
        # mus = (distris.mean - 0.5) * 2.0 * train_config['max_action']
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
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> states = torch.randn(64,111)
            >>> actions, log_probs = train_net.get_sample_data(states) 
        """
        distris = self.get_distris(states)
        sample_actions = distris.sample()
        log_probs = distris.log_prob(sample_actions)
        actions = (sample_actions - 0.5) * 2.0 * train_config['max_action']
        return sample_actions,actions,log_probs
    
    def get_check_data(self,states):
        """
        Return expectations of states, entropy of state distributions and log probilities of taking the best actions

        Args:
            states

        Example:
            >>> current_env_name = 'Ant' 
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> states = torch.randn(64,111)
            >>> mus, entropy, log_probs = train_net.get_check_data(states) 
        """
        distris = self.get_distris(states)
        mus = (distris.mean - 0.5) * 2.0 * train_config['max_action']
        log_probs = distris.log_prob(distris.mean)
        return mus,distris.entropy(),log_probs
    
    def get_value(self,states):
        """
        Calculate state values

        Args:
            states

        Return:
            state values
        
        Example:
            >>> current_env_name = 'Ant' 
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> states = torch.randn(64,111) 
            >>> train_net.get_value(states) 
        """
        return self.value(states)
    
    def sample_noise(self):
        """Add normal noise to network parameter, more details see in NoisyLinear class 

        Example:
            >>> current_env_name = 'Ant' 
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> train_net.sample_noise()
        """
        if train_envs[current_env_name].use_noise:
            for noise_layer in self.alpha_noisy_layers:
                noise_layer.sample_noise()
            for noise_layer in self.beta_noisy_layers:
                noise_layer.sample_noise()
    
class PPOMujocoBetaShareGAEUtils(AlgoBase.AlgoBaseUtils):
    pass
                                                                                                
class PPOMujocoBetaShareGAEAgent(AlgoBase.AlgoBaseAgent):
    """
    Agent class used with continues PPO, allowing collect data and evaluate agents.

    Args:
        sample_net: policy network (default: PPOMujocoNormalShareNet)
        model_dict: a dict of model configuration
        is_checker: if "True", then evaluating the agent through running 1024 timesteps with 
                    the highest probility of action, else collecting the training data.

    Example:
        >>> train_net = PPOMujocoBetaShareGAENet()
        >>> # Collecting training data
        >>> sample_agent = PPOMujocoBetaShareGAEAgent(train_net,model_dict,is_checker=False)
        >>> transition = sample_agent.sample_env()
        >>> # Evaluating agent 
        >>> check_agent = PPOMujocoBetaShareGAEAgent(train_net,model_dict,is_checker=True)
        >>> info = check_agent.check_env()
        >>> print(info['sum_rewards'], info['mean_entropys'], info['mean_mus'], info['mean_log_probs'])

    """
    def __init__(self,sample_net:PPOMujocoBetaShareGAENet,model_dict,is_checker):
        super(PPOMujocoBetaShareGAEAgent,self).__init__()
        self.sample_net = sample_net
        self.model_dict = model_dict        
        self.num_steps = train_config['num_steps']
        self.num_envs = train_config['num_envs']
        self.gae_length = train_config['gae_length']
        self.rewards = []
        
        env_name = train_envs[current_env_name].env_name
    
        if not is_checker:
            self.envs = [gym.make(env_name) for _ in range(self.num_envs)]
            self.states = [self.envs[i].reset() for i in range(self.num_envs)]
            self.exps_list = [[] for _ in range(self.num_envs)]
        else:
            print("PPOMujocoBetaShareGAE check mujoco env is",env_name)
            self.envs = gym.make(env_name)
            self.states = self.envs.reset()
            self.num_steps = 1024
            
    def get_comment_info(self):
        return current_env_name + "_" + train_config['tensorboard_comment']
        
    def sample_env(self):
        """collect training data 
        Example:
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> sample_agent = PPOMujocoBetaShareGAEAgent(train_net,model_dict,is_checker=False)
            >>> transition = sample_agent.sample_env()
        """

        while len(self.exps_list[0]) < self.num_steps:
            
            sample_acitons,actions,log_probs = self.get_sample_actions(self.states)
            for i in range(self.num_envs):
                next_state_n, reward_n, done_n, _ = self.envs[i].step(actions[i])                
                if done_n:
                    next_state_n = self.envs[i].reset()
                                    
                self.exps_list[i].append([self.states[i],sample_acitons[i],reward_n,done_n,log_probs[i],self.model_dict['train_version']])
                self.states[i] = next_state_n
                
        # Starting training
        train_exps = self.exps_list
        #删掉前self.gae_length长度
        self.exps_list = [ exps[self.gae_length:self.num_steps] for exps in self.exps_list ]
        return train_exps
    
    def check_env(self):
        """evaluate agent
        Example:
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> check_agent = PPOMujocoBetaShareGAEAgent(train_net,model_dict,is_checker=True)
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
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> sample_agent = PPOMujocoBetaShareGAEAgent(train_net,model_dict,is_checker=False)
            >>> states = torch.randn(64,111) 
            >>> actions, log_probs = sample_agent.get_sample_actions(states)
        """
        states_v = torch.Tensor(np.array(states))
        sample_acitons,actions,log_probs = self.sample_net.get_sample_data(states_v)
        return sample_acitons.cpu().numpy(), actions.cpu().numpy(),log_probs.cpu().numpy()
    
    @torch.no_grad()
    def get_check_action(self,state):
        """Get the highest probility of action, and it's entropy and log probility 
        Example:
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> check_agent = PPOMujocoNormalShareGAEAgent(train_net,model_dict,is_checker=True)
            >>> states = torch.randn(111) 
            >>> mu, entropy, log_prob = check_agent.get_check_actions(state)
        """
        state_v = torch.Tensor(np.array(state))
        mu,entropy,log_prob = self.sample_net.get_check_data(state_v)
        return mu.cpu().numpy(),entropy.cpu().numpy(),log_prob.cpu().numpy()
        
class PPOMujocoBetaShareGAECalculate(AlgoBase.AlgoBaseCalculate):
    """
    training class used with continues PPO

    Args:
        share_model: policy network (default: PPOMujocoBetaShareGAENet)
        model_dict: a dict of model configuration
        calculate_index: the :math:`calculate_index`th agent for training

    
    Example:
        >>> train_net = PPOMujocoNormalShareGAENet()
        >>> calculate = PPOMujocoNormalShareGAECalculate(train_net,model_dict,calculate_index)
        >>> # samples are from transitions
        >>> calculate.begin_batch_train(samples)
        >>> for _ in range(REPEAT_TIMES):
        >>>    calculate.generate_grads()
        >>> calculate.end_batch_train()

    """
    def __init__(self,share_model:PPOMujocoBetaShareGAENet,model_dict,calculate_index):
        super(PPOMujocoBetaShareGAECalculate,self).__init__()
        self.model_dict = model_dict
        self.share_model = share_model
        self.gae_length = train_config['gae_length']      
        self.calculate_number = self.model_dict['num_trainer']
        self.calculate_index = calculate_index
        self.train_version = 0
        
        if train_config['use_gpu'] and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_index = self.calculate_index % device_count
            self.device = torch.device('cuda',device_index)
        else:
            self.device = torch.device('cpu')
        
        self.calculate_net = PPOMujocoBetaShareGAENet()
        self.calculate_net.to(self.device)
        #self.calculate_net.load_state_dict(self.share_model.state_dict())
    
        self.share_optim = torch.optim.Adam(params=self.share_model.parameters(), lr=train_config['learning_rate'])
        
        
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.advantage_list = None
        self.returns_list = None
        
    def begin_batch_train(self, samples_list: list):   
        """store training data
        Example:
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> calculate = PPOMujocoBetaShareGAECalculate(train_net,model_dict,calculate_index)
            >>> calculate.begin_batch_train(samples)
        """  

        s_states = [np.array([s[0] for s in samples]) for samples in samples_list]
        s_actions = [np.array([s[1] for s in samples]) for samples in samples_list]
        s_log_probs = [np.array([s[4] for s in samples]) for samples in samples_list]
        
        s_rewards = [np.array([s[2] for s in samples]) for samples in samples_list]
        s_dones = [np.array([s[3] for s in samples]) for samples in samples_list]
        
        #s_versions = [s[6] for s in samples]
        
        self.states = [torch.Tensor(states).to(self.device) for states in s_states]
        self.actions = [torch.Tensor(actions).to(self.device) for actions in s_actions]
        self.old_log_probs = [torch.Tensor(log_probs).to(self.device) for log_probs in s_log_probs]
        self.rewards = s_rewards
        self.dones = s_dones
        
        self.states_list = torch.cat([states[0:self.gae_length] for states in self.states])
        self.actions_list = torch.cat([actions[0:self.gae_length] for actions in self.actions])
        self.old_log_probs_list = torch.cat([old_log_probs[0:self.gae_length] for old_log_probs in self.old_log_probs])
        
    def end_batch_train(self):
        """clear training data, update learning rate and noisy network
        Example:
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> calculate = PPOMujocoBetaShareGAECalculate(train_net,model_dict,calculate_index)
            >>> calculate.end_batch_train(samples)
        """
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.advantage_list = None
        self.returns_list = None
        
        train_version = self.model_dict[self.calculate_index]
        self.decay_lr(train_version)
        
        # Reset sampling noise
        if self.calculate_index == self.calculate_number - 1:
            self.share_model.sample_noise()
            
    def calculate_samples_gae(self):
        """Calculate samples generalized advantage estimator, moee details see in algo_base.calculate_gae
        """
        gamma = train_config['gamma']
        gae_lambda = train_config['gae_lambda']
        
        np_advantages = []
        np_returns = []
        
        for states,rewards,dones in zip(self.states,self.rewards,self.dones):
            with torch.no_grad():
                values = self.calculate_net.get_value(states)
                            
            advantages,returns = AlgoBase.calculate_gae(values.cpu().numpy().reshape(-1),rewards,dones,gamma,gae_lambda)
            np_advantages.extend(advantages[0:self.gae_length])
            np_returns.extend(returns[0:self.gae_length])
            
        return np.array(np_advantages), np.array(np_returns)
            
    def decay_lr(self, version):
        """decrease learning rate:
        :math:`lr = lr(1- ve / max_ve )`
        where :math:`lr` is learning rate, :math:`ve` is current version and :math:`max_ve` is the highest version.
        Minimum learning rate is equal to 1e-6

        Example:
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> calculate = PPOMujocoBetaShareGAECalculate(train_net,model_dict,calculate_index)
            >>> calculate.decay_lr(calculate_index)
            
        """
        if train_config['enable_lr_decay']:
            lr_now = train_config['learning_rate'] * (1 - version*1.0 / train_envs[current_env_name].max_version)
            if lr_now <= 1e-6:
                lr_now = 1e-6
            
            if self.share_optim is not None:
                for param in self.share_optim.param_groups:
                    param['lr'] = lr_now
                                                
    def generate_grads(self):
        """ update share network parameters. 
        
        If action is discrete, then :math:`ratio1 = exp(new_log_probs - old_log_probs)`, if action is continues, then
        :math:`ratio2 = \prod{ratio1}` and expand to the same dimention as :math:`ratio1`, if action is mixed, then 
        :math:`ratio3 = ratio1 * ratio_coef + ratio2 * (1.0 - ratio_coef)`, where :math:`ratio_coef` is weight coefficent.
        

        Example:
            >>> train_net = PPOMujocoBetaShareGAENet()
            >>> calculate = PPOMujocoBetaShareGAECalculate(train_net,model_dict,calculate_index)
            >>> calculate.begin_batch_train(samples)
            >>> REPEAT_TIMES = 10
            >>> for _ in range(REPEAT_TIMES):
            >>>     calculate.generate_grads()
            >>> calculate.end_batch_train()
        """ 
        train_version = self.model_dict[self.calculate_index]
        grad_norm = train_config['grad_norm']
        vf_coef = train_config['vf_coef']
        pg_loss_type = train_config['pg_loss_type']
        mini_batch_size = train_config['mini_batch_size']

        ent_coef = self.get_ent_coef(train_version)
        ratio_coef = self.get_ratio_coef(train_version)
    

        self.calculate_net.load_state_dict(self.share_model.state_dict())
                                
        #start = timer()
        np_advantages,np_returns = self.calculate_samples_gae()
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
            mini_old_log_probs = self.old_log_probs_list[start_index:end_index]
            
            self.calculate_net.load_state_dict(self.share_model.state_dict())
                
            mini_new_values,mini_new_log_probs,mini_entropys = self.calculate_net(mini_states,mini_actions)
            
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
            
            v_loss = F.mse_loss(mini_returns, mini_new_values) * vf_coef
            
            e_loss = -torch.mean(mini_entropys) * ent_coef
            
            loss = pg_loss + v_loss + e_loss

            self.calculate_net.zero_grad()

            loss.backward()
            
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
    # used for tensorboard naming
    comment = "_PPOMujocoBetaShareGAE_" + current_env_name + "_" + train_config['tensorboard_comment']
    writer = SummaryWriter(comment=comment)
    
    # initialize training network
    train_net = PPOMujocoBetaShareGAENet()

    # set model dictionary
    model_dict = {}
    model_dict[0] = 0
    model_dict['num_trainer'] = 1
    model_dict['train_version'] = 0

    # initialize a RL agent, smaple agent used for sampling training data, check agent used for evaluating 
    # and calculate used for calculating gradients
    sample_agent = PPOMujocoBetaShareGAEAgent(train_net,model_dict,is_checker=False)
    check_agent = PPOMujocoBetaShareGAEAgent(train_net,model_dict,is_checker=True)
    calculate = PPOMujocoBetaShareGAECalculate(train_net,model_dict,0)

    # hyperparameters
    MAX_VERSION = 3000
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
    
    
