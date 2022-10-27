"""
1 Writing version
2 Writing model
3 Writing experience
4 Reading version
5 Reading model
6 Reading experience
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import sys
import redis
from libs.log import Log
import zlib
import pickle
import torch.nn as nn

class RedisCache:
    """
    Building redis database for storing model information and outputing model information.

    Args:
        log: logging
        redis_config: redis parameters

    
    """
    exps_name = 'exps'  
    exit_flag_name = 'exit'
    
    sample_version_name = 'sample_version'
    sample_model_name = 'sample_model'
        
    def __init__(self,log:Log,redis_config:dict):
        self.log = log
        self.redis_config = redis_config
        self.conn = redis.Redis(host=self.redis_config['ip'], 
                                port=self.redis_config['port'],
                                db=self.redis_config['db'],
                                password=self.redis_config['pw'])
                
        if not self.conn.ping():
            self.log.log_info("redis connect fail and will exit",print_screen=True)
            exit()
        else:
            pass
 
    def __del__(self):
        """Close redis"""
        self.conn.close()
        
    
        
   
    def clear_data(self):
        """Clear redis all data"""
        self.conn.flushall()
        
    
    def clear_db(self):
        """Clear redis database"""
        self.conn.flushdb()
        
    
    def set_exit_flag(self,exit_flag):
        """Setting exit flag"""
        try:
            self.conn.set(RedisCache.exit_flag_name,int(exit_flag))
            return True
        
        except Exception:
            self.log.log_exception(print_screen=True)
            return False
    
    
    def get_exit_flag(self):
        """Getting exit flag"""
        try:
            exit_flag =  self.conn.get(RedisCache.exit_flag_name)
            if exit_flag is not None:
                return int(exit_flag)
            else:
                return None
            
        except Exception: 
            self.log.log_exception(print_screen=True)
            return None
        
    
    def set_model_state_dict_version(self,model:nn.Module,version):
        """Setting training model parameters"""
        try:    
            state_dict = model.state_dict()
            bytes_buffer = pickle.dumps(state_dict,protocol = pickle.HIGHEST_PROTOCOL)
            bytes_buffer = zlib.compress(bytes_buffer)
            self.conn.set(RedisCache.sample_model_name,bytes_buffer)
            self.conn.set(RedisCache.sample_version_name,int(version))
            return True
        
        except Exception:
            self.log.log_exception(print_screen=True)
            return False
        
    
    def get_model_version(self):
        """Getting training model version"""
        try:
            version =  self.conn.get(RedisCache.sample_version_name)
            if version is not None:
                return int(version)
            else:
                return None
            
        except Exception: 
            self.log.log_exception(print_screen=True)
            return None
            
    
    def get_model_state_dict(self,model:nn.Module):
        """Getting training model"""
        try:
            result=self.conn.get(RedisCache.sample_model_name)
            if result is not None:
                bytes_buffer = zlib.decompress(result)
                state_dict = pickle.loads(bytes_buffer)
                model.load_state_dict(state_dict)    
                return True
            else:
                return False

        except Exception:
            self.log.log_exception(print_screen=True)
            return False
                        
    
    def push_exps(self,exps,sample_version):
        """Push sampling experience"""
        try:
            exps_info = dict()
            exps_info['sample_version'] = sample_version
            exps_info['exps'] = exps
            exps_info = pickle.dumps(exps_info,protocol = pickle.HIGHEST_PROTOCOL)
            exps_info = zlib.compress(exps_info)
            self.conn.lpush(RedisCache.exps_name,exps_info)  
            return True
        
        except Exception: 
            self.log.log_exception(print_screen=True)
            return False
    
    
    def pop_exps(self):
        """Getting sampling experience"""
        try:
            # Returned as Tuple(key,value), where key is RedisCache.exps_name
            # Call in blocking mode
            exps_info = self.conn.brpop(RedisCache.exps_name)
            exps_info = zlib.decompress(exps_info[1])
            exps_info = pickle.loads(exps_info)
            return exps_info['exps'],exps_info['sample_version']      
        except Exception:
            self.log.log_exception(print_screen=True)
            return None,None
        
    
    def get_exps_length(self):
        """Getting experience length in redis"""
        try:
            length = self.conn.llen(RedisCache.exps_name)  
            return length
        
        except Exception: 
            self.log.log_exception(print_screen=True)
            return None