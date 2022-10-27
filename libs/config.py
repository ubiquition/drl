import sys,os,traceback
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

from algo_envs.algo_base import AlgoBaseNet,AlgoBaseUtils,AlgoBaseAgent,AlgoBaseCalculate

from algo_envs.ppo_microrts_hogwild import PPOMicroRTSHogwildNet,PPOMicroRTSHogwildUtils,PPOMicroRTSHogwildAgent,PPOMicroRTSHogwildCalculate
from algo_envs.ppo_microrts_share import PPOMicroRTSShareNet,PPOMicroRTSShareUtils,PPOMicroRTSShareAgent,PPOMicroRTSShareCalculate
from algo_envs.ppo_microrts_share_gae import PPOMicroRTSShareGAENet,PPOMicroRTSShareGAEUtils,PPOMicroRTSShareGAEAgent,PPOMicroRTSShareGAECalculate

from algo_envs.ppo_mujoco_beta_hogwild import PPOMujocoBetaHogwildNet,PPOMujocoBetaHogwildUtils,PPOMujocoBetaHogwildAgent,PPOMujocoBetaHogwildCalculate
from algo_envs.ppo_mujoco_beta_share import PPOMujocoBetaShareNet,PPOMujocoBetaShareUtils,PPOMujocoBetaShareAgent,PPOMujocoBetaShareCalculate
from algo_envs.ppo_mujoco_beta_share_gae import PPOMujocoBetaShareGAENet,PPOMujocoBetaShareGAEUtils,PPOMujocoBetaShareGAEAgent,PPOMujocoBetaShareGAECalculate

from algo_envs.ppo_mujoco_normal_hogwild import PPOMujocoNormalHogwildNet,PPOMujocoNormalHogwildUtils,PPOMujocoNormalHogwildAgent,PPOMujocoNormalHogwildCalculate
from algo_envs.ppo_mujoco_normal_share import PPOMujocoNormalShareNet,PPOMujocoNormalShareUtils,PPOMujocoNormalShareAgent,PPOMujocoNormalShareCalculate
from algo_envs.ppo_mujoco_normal_share_gae import PPOMujocoNormalShareGAENet,PPOMujocoNormalShareGAEUtils,PPOMujocoNormalShareGAEAgent,PPOMujocoNormalShareGAECalculate
 
def create_net(env_name) -> AlgoBaseNet:
    """Loading environment network class

    Args:
        env_name

    Example:
        >>> import libs.config as config
        >>> env_name = 'PPOMujocoNormalShareGAE'
        >>> train_net = config.create_net(env_name)
    """
    net_name = env_name + 'Net'
    try:
        return globals()[net_name]()
    except:
        print("create_net error. net name is:",net_name)
        exit()
        
def create_utils(env_name,train_net) -> AlgoBaseUtils:
    """Loading environment utils calss

    Args:
        env_name
        train_net: network structure for training

    Example:
        >>> import libs.config as config
        >>> env_name = 'PPOMujocoNormalShareGAE'
        >>> train_net = config.create_net(env_name)
        >>> utils = config.creat_utils(env_name, train_net)
    """
    utils_name = env_name + 'Utils'
    try:
        return globals()[utils_name](train_net)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error = "Exception: " + repr(traceback.format_exception(exc_type, exc_value, exc_traceback))  # Turn abnormal information to string
        print("create_utils error. utils name is:",utils_name,error)
        exit()

def create_agent(env_name,sample_net,model_dict,is_checker = False) -> AlgoBaseAgent:
    """Loading environment-agent class

    Args:
        env_name
        sample_net: training network
        model_dict: a dictionary of model training information, see train_main_local.py
        is_checker: whether evaluate agent

    Example:
        >>> import libs.config as config
        >>> env_name = 'PPOMujocoNormalShareGAE'
        >>> train_net = config.create_net(env_name)
        >>> agent = config.creat_agent(env_name, train_net, model_dict, is_check = False)
    """
    agent_name = env_name + 'Agent'
    try:
        return globals()[agent_name](sample_net,model_dict,is_checker)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error = "Exception: " + repr(traceback.format_exception(exc_type, exc_value, exc_traceback))  # Turn abnormal information to string
        if is_checker:
            print("create_check_agent error. agent name is:",agent_name,error)
        else:
            print("create_sample_agent error. agent name is:",agent_name,error)
        exit()

def create_calculate(env_name,calculate_net,model_dict,calculate_index) -> AlgoBaseCalculate:
    """Loading environment-gradient calculate calss

    Args:
        env_name
        calculate_net: training network
        model_dict: a dictionary of model training information, see train_main_local.py
        calculate_index: the number of trainer id

    Example:
        >>> import libs.config as config
        >>> env_name = 'PPOMujocoNormalShareGAE'
        >>> train_net = config.create_net(env_name)
        >>> calculate = config.creat_calculate(env_name, train_net, model_dict, 1)
    """
    calculate_name = env_name + 'Calculate'
    try:
        return globals()[calculate_name](calculate_net,model_dict,calculate_index)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error = "Exception: " + repr(traceback.format_exception(exc_type, exc_value, exc_traceback))  # Turn abnormal information to string
        print("create_calculate error. calculate name is:",calculate_name,error)
        exit()
    

