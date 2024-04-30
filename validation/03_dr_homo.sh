#! /bin/bash
#SBATCH --job-name=dr_homo                         # Job name
#SBATCH --output log/03_dr_homo_%J.log                # Output log file
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --cpus-per-task=4         # 单任务使用的 CPU 核心数为 4
set -eux

#list_num=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})

method=$1
shift
save=1

sub_list_file=../sub_lists/sub_list_drt_rt.txt  # out-of-sample

rfMRI_dir=/data0/user/cyli/HCP/HCP_retest/rfmri
dti_dir=/data0/user/cyli/HCP/HCP_retest/dmri
indiv_dir=../indiv_atlas/HCP1200

matlab  -nodisplay -nosplash -r "calc_func_anat_homo_pair('$method','$rfMRI_dir','$dti_dir','$indiv_dir', '$sub_list_file', $save);exit"
