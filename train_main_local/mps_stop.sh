#!/bin/bash
echo quit | nvidia-cuda-mps-control
nvidia-smi -i 0 -c DEFAULT
nvidia-smi -i 1 -c DEFAULT
nvidia-smi -i 2 -c DEFAULT
nvidia-smi -i 3 -c DEFAULT
nvidia-smi -i 4 -c DEFAULT
nvidia-smi -i 5 -c DEFAULT
nvidia-smi -i 6 -c DEFAULT
nvidia-smi -i 7 -c DEFAULT