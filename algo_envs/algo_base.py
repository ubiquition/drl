import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.distributions.categorical import Categorical
import numba

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization (default scaling factor = :math:`\sqrt(2)`), bias = 0
    """

    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

@numba.njit()
def calculate_gae(values,rewards,dones,gamma,gae_lambda):
    """Calculate generalized advantage estimation (GAE):
    :math:`GAE(\gamma, \lambda) = \sum_{l= 0} ^{infinite} (\gamma\lambda)^l\delta_{t+l}`,
    where :math:`\gamma, \lambda` are discount factor and exponentially-weighted average coefficient.
    More details in https://arxiv.org/abs/1506.02438.

    Args:
        values: a batch of state-values
        rewards: a batch of rewards with respect to a GAE length
        dones: If done, then dones = True, else dones = False
        gamma: discount factor
        gae_lambda: exponentially-weighted average coefficient


    Example:
        >>> states = np.random.randn(256, 128) # batch size = 256, trajectory = 128
        >>> rewards = np.random.randn(256, 128)
        >>> dones = [False for _ in range(128)]
        >>> dones[-1] = True
        >>> gamma = 0.95
        >>> gae_lambda = 0.5
        >>> values = V(torch.Tensor(states))
        >>> import algo_base
        >>> algobase.calculate_gae(values.cpu().numpy().reshape(-1), rewards,dones, gamma, gae_lambda)
    """
    if len(values.shape) != 1:
        return None,None
    
    length = values.shape[0]
    
    # Ensuring dones[length-1] = True
    dones[length-1] = True
    
    # if not dones[length-1]:
    #     return None,None
    
    # # Ensuring values shape = (length,)
    # if type(values) is not np.ndarray or values.shape != (length,):
    #     return None,None
    
    # # rewards shape = (length,)
    # if type(rewards) is not np.ndarray or rewards.shape != (length,):
    #     return None,None
    
    # #dones shape = (length,)
    # if type(dones) is not np.ndarray or dones.shape != (length,):
    #     return None,None
    
    advantages = np.zeros(length, dtype=np.float32)
    returns = np.zeros(length, dtype=np.float32)
    
    # Calculating advantage value
    last_gae = 0.0
    for index in range(length-1,-1,-1):
        if dones[index]:
            delta = rewards[index] - values[index]
            last_gae = delta
        else:
            delta = rewards[index] + gamma * values[index+1] - values[index]
            last_gae = delta + gamma * gae_lambda * last_gae
            
        advantages[index] = last_gae
        returns[index] = last_gae + values[index]
                                         
    return advantages, returns
    
class NoisyLinear(nn.Linear):
    """Applies a noisy linear transformation to the incoming data: :math:`y = x(0.017*\mu)^T + (0.017*\mu)`,
    where :math:`\mu` is sampled from uniform :math:`N(-\sqrt{3 / in_features}, \sqrt{3 / in_features})`, more details in
    https://arxiv.org/abs/1706.10295.


    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Examples:

        >>> m = nn.NoisyLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, in_features, out_features,sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)
        
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
            
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        
        if not self.training:
            return super(NoisyLinear, self).forward(input)
        
        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * self.epsilon_bias.data
            
        v = self.weight + self.sigma_weight * self.epsilon_weight.data
        return F.linear(input, v, bias)

    def sample_noise(self):
        self.epsilon_weight.normal_()
        if self.bias is not None:
            self.epsilon_bias.normal_()
            
class MaskedCategorical:
    """Creates a masked categorical distribution parameterized by `probs`.
    Args:
        probs: probilities

    Example:
        >>> m = MaskedCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor(3)
    """
    def __init__(self, probs):
        self.origin_probs = probs
        self.probs = F.softmax(probs,dim=-1)
        self.dist = Categorical(self.probs)
            
    def update_masks(self,masks,device = 'cpu'):
        """Mask some actions which should not be considered.
        
        Args:
            masks

        Example:
            >>> m = MaskedCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
            >>> mask = torch.tensor([0, 1, 0, 1])
            >>> m.update_masks(masks = mask)
        """
        if masks is None:
            return self
        # Using torch.lerp() is faster than torch.where() method.
        probs = torch.lerp(self.origin_probs, torch.tensor(-1e+8).to(device), 1.0 - masks)
        self.probs = F.softmax(probs,dim=-1)
        self.dist = Categorical(self.probs)
        return self
    
    def sample(self):
        """Random sample one action.
        
        """
        #actions = torch.multinomial(self.probs, num_samples=1, replacement=True)[:, 0]
        actions = self.dist.sample()
        return actions
    
    def log_prob(self,actions):
        """Calculate the log probilities of actions
        
        Example:
            >>> m = MaskedCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
            >>> action = m.sample()
            >>> log_prob = m.log_prob(action)
        """
        return self.dist.log_prob(actions)
    
    def entropy(self):
        """Calculate the entropy of the distribution
        
        Example:
            >>> m = MaskedCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
            >>> entropy = m.entropy()
        """
        return self.dist.entropy()
    
    def argmax(self):
        """Return the highest probility index
        
        Example:
            >>> m = MaskedCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
            >>> max_index = m.argmax()
        """
        return torch.argmax(self.probs,dim=-1)
    
    def argmin(self):
        """Return the lowest probility index
        
        Example:
            >>> m = MaskedCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
            >>> max_index = m.argmin()
        """
        return torch.argmin(self.probs,dim=-1)
                  
class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model:nn.Module):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)
        
class AlgoBaseNet(nn.Module):
    """Base class for all `Net` neural network classes,
        these kind of models should also subclass this class.
    """
    def __init__(self):
        super(AlgoBaseNet,self).__init__()
                  
    def forward(self,states):
        raise NotImplementedError

  
class AlgoBaseUtils:
    """Base class for all `Utils` classes, these kind of models should also subclass this class.
    """
    def __init__(self):
        pass
    
    #state_dict
    def save_model(self):
        pass
    
    def load_model(self):
        pass
    
    
    def get_bytes_buffer_from_model(self):
        pass
    
    def get_model_from_bytes_buffer(self,bytes_buffer):
        pass
    
    # onnx interface  
    # def get_onnx_buffer_from_model(self):
    #     pass
    
    # def get_session_from_onnx_buffer(self,bytes_buffer):
    #     pass
    
    
    def create_target_net(self):
        # target_net = copy.deepcopy(self.train_net)
        # target_net.to(device)
        # target_net.load_state_dict(self.state_dict())
        # target_net.share_memory()
        # return target_net
        return None
    
    
    def create_optimizer_net(self):
        # optimizer_net = copy.deepcopy(self.train_net)
        # optimizer_net.to(device)
        # optimizer_net.load_state_dict(self.state_dict())
        # return optimizer_net
        return None
    
    # Return a bool type indicating whether an updated version is required   
    def update_state(self,grads_buffer,version,optimizer_net):
        raise NotImplementedError
    
    # Return the bool type, indicating whether to update the version, and noise net, external synchronization model to redis
    def update_version(self,version):
        raise NotImplementedError
        

class AlgoBaseAgent:
    """Base class for all `Agent` classes, these kind of models should also subclass this class.
    """
    def __init__(self):
        pass
    
    def sample_env(self):
        raise NotImplementedError
    
    def check_env(self):
        raise NotImplementedError
    
    
    def save_policy(self):
        pass
    
    
    def get_comment_info(self):
        return "AlgoBaseAgent"
        
class AlgoBaseCalculate:
    """Base class for all `Calculate` classes, these kind of models should also subclass this class.
    """
    def __init__(self):
        pass
    
    def set_grads_queue(self,grads_queue):
        pass
    
    def begin_batch_train(self,samples:list):
        pass
    
    def generate_grads(self):
        raise NotImplementedError
    
    def end_batch_train(self):
        pass
        
class GradCoef(autograd.Function):       
    # Model forward
    @staticmethod
    def forward(ctx, x, coeff):
        # Save coeff as a member variable of ctx         
        ctx.coeff = coeff
        return x.view_as(x)

    # Model gradient backpropagation
    @staticmethod
    def backward(ctx, grad_output):
        # The number of outputs of backward should be the same as the number of inputs of forward. 
        # Here coeff does not need gradient, so it returns None   
        return ctx.coeff * grad_output, None
