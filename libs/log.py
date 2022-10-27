"""
log system The content storage path is ../logs
"""
import os
import time
import traceback, sys

class Log:
    def __init__(self,dir_name):
        """
        Check if there is a corresponding file directory under `../logs`, if not, create it.
        """
        self.dir_name = os.path.join(os.path.abspath(os.path.dirname(__file__)),'../logs/',dir_name)#"../logs/" + dir_name
        try:
            if not os.path.exists(self.dir_name):
                os.makedirs(self.dir_name)
        except:
            print("create log dir: "+ self.dir_name + " error")
    
    def log_info(self,message:str,print_screen:bool=False):
        # Detect the files
        file_name = self.dir_name + "/" + time.strftime("%Y-%m-%d", time.localtime()) + ".log"
        message = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + message
        
        if print_screen:
            print(message)
            
        message = message + "\n"
        try:
            fa = open(file_name,"a")
            fa.write(message)
        except:
            print("write log message: "+ message + " error")
        finally:
            fa.close()

    # Exception log
    def log_exception(self,print_screen:bool=False):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error = "Exception: " + repr(traceback.format_exception(exc_type, exc_value, exc_traceback))  # Convert exception information to string
        self.log_info(error,print_screen)
    