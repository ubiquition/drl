
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch.multiprocessing as mp
import time
import libs.log as log
import libs.config as config
import queue
import libs.utils as utils
import libs.redis_config as redis_config
import libs.redis_cache as redis_cache

class Trainer:
 
    """
    For training agents

    Args:
        id: instance tag
        model_dict: a dictionary for storing network parameters, see more in train_main_local.py file
        share_model: network structure
        sample_batch: a batch size
        repeat_times: training times for each batch
        env_name: environment class name
        log: logging

    Example:
        >>> import libs.config as config
        >>> env_name = PPOMujocoNormalShareGAE
        >>> train_net = config.create_net(env_name)  
        >>> samplers= sampler.Sampler(id = 0, model_dict=model_dict,share_model=train_net, env_name=model_env_name,log=train_log)
        >>> samplers.run_sampler(train_seed = train_seed)
    """
 
  
    def __init__(self, id,model_dict,share_model,sample_batch,repeat_times,env_name,log:log.Log):
        self.trainer_id = id
        self.model_dict = model_dict
        self.share_model = share_model
        self.sample_batch = sample_batch
        self.repeat_times = repeat_times
        self.env_name = env_name
        self.process = None
        self.log = log
            
       
    def process_function(self):
        """
        Creating calculate class and experience redis prepare for training
        """
        # Setting random seed
        utils.setup_seed(self.train_seed)
                                
        calculate = config.create_calculate(self.env_name,self.share_model,self.model_dict,self.trainer_id)
        exps_redis_config = redis_config.get_current_redis_exps_config()
        exps_redis_cache = redis_cache.RedisCache(self.log,exps_redis_config)
        
        samples_list = []
                
        while True:
            if self.model_dict['is_exit']:
                break
            try:
                # 1 pop samples， 2 calculate gradients
                samples_item,exps_version = exps_redis_cache.pop_exps()
                if samples_item is not None:
                    samples_list.append(samples_item)

                if len(samples_list) == self.sample_batch:                    
                    start_time = time.time()
                    
                    train_version = self.model_dict['train_version']
                    
                    calculate.begin_batch_train(samples_list)
                    for _ in range(self.repeat_times):
                        calculate.generate_grads()
                    calculate.end_batch_train()
                    
                    end_time = time.time()-start_time                    
                    if self.trainer_id == 0:
                        print('calculate_time:',end_time,"train_version:",train_version,"exps_version:",exps_version)
                                                
                    samples_list = []
                    
                    # Cumulative training times
                    self.model_dict[self.trainer_id] = self.model_dict[self.trainer_id] + 1
                                    
            except Exception:
                self.log.log_exception(print_screen=True)
                continue
            
        # Exit
        try:
            del exps_redis_cache
        except:
            self.log.log_exception(print_screen=True)
        
        self.log.log_info('exit trainer processid ' + str(self.process.pid) + " trainerid " + str(self.trainer_id),print_screen=True)
                
    def run_trainer(self,train_seed):
        """
        Starting training
        """
        self.train_seed = train_seed
        self.process = mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info('start trainer processid ' + str(self.process.pid) + " trainerid " + str(self.trainer_id),print_screen=True)
        
    def stop(self):
        """Stop training
        """
        try:
            if self.process is not None:
                self.process.terminate()
                self.process.join()
        except:
            self.log.log_exception(print_screen=True)
        