import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))


import torch.multiprocessing as mp
import torch.nn as nn

import time
import libs.log as log
import libs.config as config
import libs.utils as utils
import libs.redis_config as redis_config
import libs.redis_cache as redis_cache

class Sampler:
    """
    For sampling training data

    Args:
        id: instance tag
        model_dict: a dictionary for storing network parameters, see more in train_main_local.py file
        share_model: network structure
        env_name: environment class name
        log: logging

    Example:
        >>> import libs.config as config
        >>> env_name = PPOMujocoNormalShareGAE
        >>> train_net = config.create_net(env_name)  
        >>> samplers= sampler.Sampler(id = 0, model_dict=model_dict,share_model=train_net, env_name=model_env_name,log=train_log)
        >>> samplers.run_sampler(train_seed = train_seed)
    """

    
    def __init__(self, id,model_dict,share_model:nn.Module,env_name,log:log.Log):
        self.sampler_id = id
        self.model_dict = model_dict
        self.share_model = share_model
        self.env_name = env_name
        self.log = log
        self.process=None
        self.sample_version = self.model_dict['sample_version']
 
    def process_function(self):
        """
        Creating agent and redis prepare for sampling
        """
        # Setting random seed
        utils.setup_seed(self.train_seed)
        # sample_net = config.create_net(self.env_name)
        # sample_net.load_state_dict(self.share_model.state_dict())
        # Ensure the model is latest
        sample_agent = config.create_agent(self.env_name,self.share_model,self.model_dict)
        
        exps_redis_config = redis_config.get_current_redis_exps_config()
        exps_redis_cache = redis_cache.RedisCache(self.log,exps_redis_config)
                
        while True:
            if self.model_dict['is_exit']:
                break
            
            self.sample_version = self.model_dict['sample_version']
            
            #if sample_version > self.sample_version:
                #sample_net.load_state_dict(self.share_model.state_dict())
            #    self.sample_version = sample_version
                
            try:
                start_time = time.time()             
                exps_list = sample_agent.sample_env()
                end_time = time.time()-start_time
                sample_version = self.model_dict['sample_version']
                if self.sampler_id == 0:
                    print('sample_time:',end_time,"begin_sample_version:",self.sample_version,"end_sample_version",sample_version)
                
                for exps in exps_list:
                    push_result = exps_redis_cache.push_exps(exps,sample_version)
                    
                    if not push_result:
                        self.log.log_info("send exps to redis failure",print_screen=True)
    
            except:
                self.log.log_exception(print_screen=True)
                continue
            
        # Ensure exit environment
        try:
            del sample_agent
        except:
            self.log.log_exception(print_screen=True)
        
        self.log.log_info('exit sampler processid ' + str(self.process.pid) + " samplerid " + str(self.sampler_id),print_screen=True)
                
    def run_sampler(self,train_seed):
        """Starting sampling training data
        """
        self.train_seed = train_seed
        self.process=mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info('start sampler processid ' + str(self.process.pid) + " samplerid " + str(self.sampler_id),print_screen=True)

    def stop(self):
        """Stop sampling
        """
        try:
            if self.process is not None:
                self.process.terminate()
                self.process.join()
        except:
            self.log.log_exception(print_screen=True)
