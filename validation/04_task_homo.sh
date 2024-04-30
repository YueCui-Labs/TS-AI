#! /bin/bash
#SBATCH --job-name=task                         # Job name
#SBATCH --output log/04_task_homo_%J.log                # Output log file
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --cpus-per-task=4         # 单任务使用的 CPU 核心数为 4
set -eux

method=$1
shift

sub_list_file=../sub_lists/sub_list_drt_rt.txt

indiv_dir=../indiv_atlas/HCP1200
tfmri_dir=/data0/user/cyli/HCP/HCP_retest/tfmri

cifti_vert_LR_file=cifti_vert_LR.mat
output_dir=result

matlab  -nodisplay -nosplash -r "task_homo('${sub_list_file}', '${indiv_dir}', '${tfmri_dir}', '${method}','${cifti_vert_LR_file}', '${output_dir}');exit"
