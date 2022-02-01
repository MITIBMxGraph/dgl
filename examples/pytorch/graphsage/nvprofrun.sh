#!/bin/bash

# script to profile distdgl

echo "Make tmp directories on node."
mkdir /state/partition1/user/$USER

echo "Launching single node training."
nvprof --profile-child-processes --unified-memory-profiling off -f -s -o /state/partition1/user/$USER/salientdgl%p.nvvp /home/gridsan/$USER/.conda/envs/salientdgl/bin/python salient_dataloader.py --data-cpu

echo "Mv files from tmp directories on nodes to dir accessible by all nodes & removing tmp dir." 
mv /state/partition1/user/$USER/* /home/gridsan/$USER/nvidia_profiling
rm -r /state/partition1/user/$USER
