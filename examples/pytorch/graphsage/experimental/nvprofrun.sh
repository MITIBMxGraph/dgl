#!/bin/bash

# script to profile distdgl

echo "Make tmp directories on nodes."
while read ip; do
  ssh -n $USER@$ip "mkdir /state/partition1/user/$USER;"
done <ip_config.txt

echo "Launching distributed training."
python3 ~/dgl/tools/launch.py --ssh_username=$USER --workspace ~/dgl/examples/pytorch/graphsage/experimental/ --num_trainers 1 --num_samplers 1 --num_servers 1 --num_omp_threads 40 --part_config ogbn-products-k2-data/ogb-product.json --ip_config ip_config.txt  "source /etc/profile; module load cuda/11.3; nvprof --profile-child-processes --unified-memory-profiling off -f -s -o /state/partition1/user/$USER/distdgl%p.nvvp /home/gridsan/$USER/.conda/envs/salientdgl/bin/python train_dist_nvtx.py --graph_name ogb-product --ip_config ip_config.txt --num_epochs 1  --batch_size 1024 --fan_out 15,10,5 --num_gpus 1"
#python3 ~/dgl/tools/launch.py --ssh_username=$USER --workspace ~/dgl/examples/pytorch/graphsage/experimental/ --num_trainers 1 --num_samplers 0 --num_servers 1 --num_omp_threads 40 --part_config ogbn-products-k2-data/ogb-product.json --ip_config ip_config.txt  "source /etc/profile; module load cuda/11.3; nvprof --profile-child-processes --unified-memory-profiling off -f -s -o /state/partition1/user/$USER/distdgl%p.nvvp /home/gridsan/$USER/.conda/envs/salientdgl/bin/python train_dist_nvtx.py --graph_name ogb-product --ip_config ip_config.txt --num_epochs 1  --batch_size 1024 --fan_out 15,10,5 --num_gpus 1"
#python3 ~/dgl/tools/launch.py --ssh_username=$USER --workspace ~/dgl/examples/pytorch/graphsage/experimental/ --num_trainers 1 --num_samplers 0 --num_servers 1 --num_omp_threads 40 --part_config ogbn-papers-k2-data/ogb-paper100M.json --ip_config ip_config.txt  "source /etc/profile; module load cuda/11.3; nvprof --profile-child-processes --unified-memory-profiling off -f -s -o /state/partition1/user/$USER/distdgl%p.nvvp /home/gridsan/$USER/.conda/envs/salientdgl/bin/python train_dist_nvtx.py --graph_name ogb-paper100M --ip_config ip_config.txt --num_epochs 1  --batch_size 1024 --fan_out 15,10,5 --num_gpus 1"

echo "Scp-ing files from tmp directories on nodes to dir accessible by all nodes & removing tmp dir." 
while read ip; do
  scp -r $USER@$ip:/state/partition1/user/$USER/* /home/gridsan/$USER/nvidia_profiling
  ssh -n $USER@$ip "rm -r /state/partition1/user/$USER"
done <ip_config.txt
