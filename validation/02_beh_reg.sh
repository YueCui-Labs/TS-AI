#! /bin/bash
#SBATCH --job-name=beh                        # Job name
#SBATCH --output log/02_beh_%J.log                # Output log file
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --cpus-per-task=4         # 单任务使用的 CPU 核心数为 4
set -eux

method=$1

sub_list_file1=../sub_lists/sub_list_drt_fold1.txt
sub_list_file2=../sub_lists/sub_list_drt_fold2.txt

matlab  -nodisplay -nosplash -r "beh_reg_2model('${method}', '${sub_list_file1}', '${sub_list_file2}');exit"
#matlab  -nodisplay -nosplash -r "beh_reg_nok_2model('${method}', '${sub_list_file1}', '${sub_list_file2}');exit"

