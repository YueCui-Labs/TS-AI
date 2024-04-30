#! /bin/bash
#SBATCH --job-name=trt                        # Job name
#SBATCH --output log/01_trt_%J.log                # Output log file
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --cpus-per-task=4         # 单任务使用的 CPU 核心数为 4
set -eux

method=$1

sub_list1=../sub_lists/sub_list_drt_fold1_rt39.txt
sub_list2=../sub_lists/sub_list_drt_fold2_rt39.txt

matlab  -nodisplay -nosplash -r "test_retest_2model('$method', '$sub_list1', '$sub_list2');exit"
