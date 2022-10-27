import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

    

# Model redis configuration information
def get_current_redis_model_config():
    return redis_args_dict_local_model

# Redis configuration information for storing experience
def get_current_redis_exps_config():
    return redis_args_dict_local_exps

#redis parameters
redis_args_dict_local_exps = dict()
redis_args_dict_local_exps['ip'] = '127.0.0.1' # Server ip
redis_args_dict_local_exps['port'] = '6379' # Server port
redis_args_dict_local_exps['db'] = '0' # Server database
redis_args_dict_local_exps['pw'] = '123456' # Server password

#redis parameters
redis_args_dict_local_model = dict()
redis_args_dict_local_model['ip'] = '127.0.0.1' # Server ip
redis_args_dict_local_model['port'] = '6379' # Server port
redis_args_dict_local_model['db'] = '1' # Server database
redis_args_dict_local_model['pw'] = '123456' # Server password