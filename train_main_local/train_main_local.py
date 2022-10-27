
"""
0 Starting logger,tensorboardX
1 grads data structure
2 Starting Worker
3 Starting Master
4 Starting redis or sampler
while True
    5 Exiting detection
        join each process
    6 Getting samples from redis or sample queue 
    7 Sending sample data to master
    8 Calculating gradients and sending updated network to master and redis

9 Clean up

"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import argparse
from torch.utils.tensorboard import SummaryWriter
import trainer
import torch.multiprocessing as mp
import libs.config as config
import libs.utils as utils
import libs.log as log
import sampler
import checker
import libs.redis_config as redis_config
import libs.redis_cache as redis_cache

# Getting environment name
def get_current_env_name():    
    #return "PPOMicroRTSHogwild"
    #return "PPOMicroRTSShare"
    #return "PPOMicroRTSShareGAE"
    
    #return "PPOMujocoBetaHogwild"
    #return "PPOMujocoBetaShare"
    #return "PPOMujocoBetaShareGAE"
    
    #return "PPOMujocoNormalHogwild"
    #return "PPOMujocoNormalShare"
    return "PPOMujocoNormalShareGAE"

# Pay attention to the configuration of redis, ensure that database should not 
# conflict at each training.

# The shorter the length of the trajectory, the faster the model is updated, but the shorter 
# the length of the trajectory, the less accurate the GAE calculation. The recommended length is 512.

# Another algorithm, for example, GAE length is 512, but only the front 256 is calculated. The two implementation methods,
# one requires the data of the first 256 for calculating, and the other sampler sends part of the data repeatly. Ensure the efficiency of GAE computing and model update.

# Mini-batch size is related to the environment. If the environment is simply, selecting for small mini-batch is better, such as 64. 
# In the complex environment, it is recommended to increase the size of the batch instead of using mini-batch.


args_dict = dict()
# Setting one sample length which can not less than two times the number of sampler multiplier the number of environment
args_dict['len_sample_queue'] = 4096

# Whether to evaluate agent
args_dict['enable_checker'] = True

# Number of trainers
args_dict['trainer_number'] = 1 

# Number of samplers
args_dict['sampler_number'] = 2

# Sample batch size
args_dict['sample_batch'] = 32 

# Training times for each batch
args_dict['repeat_times'] = 2 

# Improve the frequency of model updates to ensure that there are various versions of model sampling samples in a trajectory, which is conducive to training.

    
if __name__ == '__main__':
    
    # Setting multi-process mode
    mp.set_start_method('spawn')
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
     
    # Getting environment parameters
    parser = argparse.ArgumentParser()
    #parser.add_argument("--trainer", type=int, required=False,default=args_dict['trainer_number'],help="trainer_number")
    #parser.add_argument("--sampler", type=int, required=False,default=args_dict['sampler_number'],help="sampler_number")
    #args = parser.parse_args()
    
    train_seed = 20121022
    
    # Setting random seed
    utils.setup_seed(train_seed)
    
    # Starting the log
    train_log = log.Log("train_main_local")

    # Current environment name for training
    model_env_name = get_current_env_name()

    # Checking environment and printing
    train_log.log_info("current train algo_env is " + model_env_name,print_screen=True)

    model_prefix= "train_main_local"    
    model_version = None
        
    # Creating network
    train_net = config.create_net(model_env_name)    
    #train_net = train_net.to(train_net.get_device()) #sampling
    # print(train_net)
    # import numpy as np
    # parameters = sum([np.prod(p.shape) for p in train_net.parameters()])
    # print(train_net,parameters)

    # Sharing network parameters
    train_net.share_memory()
        
    #train_utils，# Used for model update, storage
    #train_utils = config.create_utils(model_env_name,train_net)
    
    # Loading model data
    current_train_version = utils.get_model_from_file(train_net,model_prefix + "_" + model_env_name,model_version)
    
    # If not yet trained, then set current train version to 0
    if current_train_version is None:
        train_log.log_info("has no model data and starts a new train",print_screen=True)
        current_train_version = 0
        
    # Current network parameter version
    model_dict = mp.Manager().dict()

    # Training network version
    model_dict['train_version'] = current_train_version 

    # Sampling network version
    model_dict['sample_version'] = current_train_version 
    
    # Exit signal
    model_dict['is_exit'] = False

    # The number of trainers
    model_dict['num_trainer']  = args_dict['trainer_number']
                    
    # Adding target net
    #target_net = train_utils.create_target_net()
    
    # Adding optimizer net 
    #optimizer_net = train_utils.create_optimizer_net()
    
    # Model redis
    model_redis_config = redis_config.get_current_redis_model_config()
    model_redis_cache = redis_cache.RedisCache(train_log,model_redis_config)
    model_redis_cache.clear_data()
    
    # Experience redis
    exps_redis_config = redis_config.get_current_redis_exps_config()
    exps_redis_cache = redis_cache.RedisCache(train_log,exps_redis_config)
    exps_redis_cache.clear_data()

    # Training containers
    trainers = []
    samplers = []
    
    # Starting trainer
    for i in range(args_dict['trainer_number']):
        l_trainer = trainer.Trainer(id=i, model_dict=model_dict,share_model=train_net,
                                    sample_batch = args_dict['sample_batch'],repeat_times = args_dict['repeat_times'],
                                    env_name=model_env_name,log=train_log)
        trainers.append(l_trainer)
        l_trainer.run_trainer(train_seed = train_seed)

    # Starting sampler
    for i in range(args_dict['sampler_number']): 
        l_sampler = sampler.Sampler(id=i, model_dict=model_dict,share_model=train_net,
                                    env_name=model_env_name,log=train_log)
        samplers.append(l_sampler)
        l_sampler.run_sampler(train_seed = train_seed)
    
    # Starting checker
    if  args_dict['enable_checker']:
        train_checker = checker.Checker(model_dict=model_dict,share_model=train_net, env_name=model_env_name,log=train_log)
        train_checker.run_checker(model_env_name+'_train_seed_'+ str(train_seed))
                
    train_log.log_info("start run train_main_local main process id is " + str(os.getpid()),print_screen=True)
    
    # Training counter
    for i in range(args_dict['trainer_number']):
        model_dict[i] = 0
    
    # Cumulative number of training
    last_trainer_version = 0
                            
    while True:
        try:
            # Exiting detection
            if utils.exit_run():
                train_log.log_info("start exit train_main_local",print_screen=True)
                
                # Saving current model version
                utils.save_model_to_file(train_net,model_prefix + "_" + model_env_name,current_train_version)

                model_dict['is_exit'] = True
                
                # Waiting process exit
                for l_sampler in samplers:
                    l_sampler.stop()
                    
                for l_trainer in trainers:
                    l_trainer.stop()
                
                if  args_dict['enable_checker']:
                    train_checker.stop()
                    
                # Clearing redis
                model_redis_cache.clear_data()
                exps_redis_cache.clear_data()
                                
                del model_redis_cache
                del exps_redis_cache
                                              
                train_log.log_info("end exit train_main_local",print_screen=True)
                break
            
            #print("sample_queue size is:",str(sample_queue.qsize()))
            
            if exps_redis_cache.get_exps_length() >= args_dict['len_sample_queue']: 
                print("exps_redis_cache is full")

                # Clearing old data
                exps_redis_cache.clear_db()
                                                
            # Ending batch train
            trainer_version = model_dict[0]
            if trainer_version > last_trainer_version:
                last_trainer_version = trainer_version
                # Updating version
                current_train_version = current_train_version + 1
                model_dict['train_version'] = current_train_version 
                model_dict['sample_version'] = current_train_version
                
                #bytes_buffer = train_utils.get_bytes_buffer_from_model()
                model_redis_cache.set_model_state_dict_version(train_net,current_train_version)
                
                # Clearing old data
                #exps_redis_cache.clear_data()
                                                        
        except:
            train_log.log_exception(print_screen=True)
    
    train_log.log_info("exit OK",print_screen=True)
