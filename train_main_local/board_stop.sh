#/bin/bash
kill -9 `ps -ef | grep tensorboard | grep python3 | grep -E "[0-9]+" -o | head -n 1`