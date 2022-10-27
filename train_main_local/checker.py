import sys,os
from xml.etree.ElementTree import Comment
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))


import torch.multiprocessing as mp

import time
import libs.config as config
from torch.utils.tensorboard import SummaryWriter
import libs.utils as utils
import libs.log as log

class Checker:
    """
    For evaluating agents

    Args:
        model_dict: a dictionary for storing network parameters, see more in train_main_local.py file
        share_model: network structure
        env_name: environment class name
        log: logging

    Example:
        >>> import libs.config as config
        >>> env_name = PPOMujocoNormalShareGAE
        >>> train_net = config.create_net(env_name)  
        >>> train_checker = checker.Checker(model_dict=model_dict,share_model=train_net, env_name=model_env_name,log=train_log)
        >>> train_checker.run_checker(model_env_name+'_train_seed_'+ str(train_seed))
    """
    def __init__(self, model_dict,share_model,env_name,log:log.Log):
    
        self.model_dict = model_dict
        self.share_model = share_model
        self.check_version = model_dict['sample_version']
        self.env_name = env_name
        self.log = log
        self.process = None
        self.comment = "checker"
 
    def process_function(self):
        """
        Creating network and agent prepare for training
        """
        # Setting random seed
        utils.setup_seed()

        check_net = config.create_net(self.env_name)
        check_net.load_state_dict(self.share_model.state_dict())    
        check_net.eval()

        check_agent = config.create_agent(self.env_name,check_net,self.model_dict,is_checker=True)
        comment = "_" + self.comment + "_" + check_agent.get_comment_info()
        writer = SummaryWriter(comment=comment)
        
        while True:
            if self.model_dict['is_exit']:
                break
            
            check_vesion = self.model_dict['sample_version']
            
            # Update current check version
            if check_vesion > self.check_version:
                self.check_version = check_vesion
                check_net.load_state_dict(self.share_model.state_dict())    
                
            try:   
                infos = check_agent.check_env()
                                    
                if isinstance(infos,dict):
                    for (key,value) in  infos.items():
                        writer.add_scalar(key, value, self.check_version)
                        
                # Save model
                #check_agent.save_policy()
                                                                    
                time.sleep(0)
            except:
                self.log.log_exception(print_screen=True)
                continue
            
        writer.close()
        
        self.log.log_info('exit checker processid ' + str(self.process.pid),print_screen=True)
                
    def run_checker(self,comment=None):
        """
        Starting evaluate agents
        """
        if comment is not None:   
            self.comment = comment
        self.process=mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info('start checker processid ' + str(self.process.pid),print_screen=True)

    def stop(self):
        """
        Stop checking
        """
        try:
            if self.process is not None:
                self.process.terminate()
                self.process.join()
        except:
            self.log.log_exception(print_screen=True)