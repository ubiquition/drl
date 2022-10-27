#/bin/bash
kill -9 `ps -ef | grep train_main_local | grep python3 | grep -E "[0-9]+" -o | head -n 1`
ps -ef | grep multiprocessing | grep -v grep | awk '{print "kill -9 "$2}' | sh