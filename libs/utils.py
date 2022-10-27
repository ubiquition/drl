import random
import numpy as np
import torch
import torch.nn as nn
import os,time


start_time = time.time()

# Exit
def exit_run(): 
    
    #if time.time()-start_time>4*60*60: return True
    
    path = os.path.abspath(os.path.dirname(__file__) + '/' + '../') 
                                   
    if os.path.exists(path+"/exit.cmd"):
        return True
    else:
        return False
    
# Loading the model from the Models folder
def get_model_from_file(model:nn.Module,prefix:str,version):
    if version is None:
        return None
    
    dir_name = os.path.join(os.path.abspath(os.path.dirname(__file__)),'../models/',prefix) +"/"

    file_name = dir_name + prefix + '_' + str(version) + ".td"
    if not os.path.isfile(file_name):
       return None
        
    model_info = torch.load(file_name)
    model.load_state_dict(model_info['state_dict'])
    return model_info['version']
    
# Saving the model to the local 
def save_model_to_file(model:nn.Module,prefix:str,version):    
    dir_name = os.path.join(os.path.abspath(os.path.dirname(__file__)),'../models/',prefix) +"/"
    
    # Determining whether there is a path
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    file_name = dir_name + prefix + '_' + str(version) + ".td"
    
    model_info = dict()
    model_info['version'] = version
    model_info['state_dict'] = model.state_dict()
            
    torch.save(model_info, file_name)
    
# Killing processes
def kill_process(pid):
    """This function is used to suspend the process corresponding to the PID.
    """
    
    if os.name == 'nt':
        # Windows system
        cmd = 'taskkill /pid ' + str(pid) + ' /f'
        try:
            os.system(cmd)
            print(pid, 'killed')
        except Exception as e:
            print(e)
    elif os.name == 'posix':
        # Linux system
        cmd = 'kill ' + str(pid)
        try:
            os.system(cmd)
            print(pid, 'killed')
        except Exception as e:
            print(e)
    else:
        print('Undefined os.name')

# Setting random seed      
def setup_seed(seed = None):
    if seed is None:
        seed = 1970010101

    #seed = 19810407
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
