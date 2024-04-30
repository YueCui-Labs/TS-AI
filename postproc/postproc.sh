#! /bin/bash
#SBATCH --job-name=post                         # Job name
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --cpus-per-task=4         # 单任务使用的 CPU 核心数为 4

sub_list=$1
shift
label_dir=$1
shift
method=$1
shift
hemi=$1

python postproc.py $sub_list $label_dir $method $hemi
