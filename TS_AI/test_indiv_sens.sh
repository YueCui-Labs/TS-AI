#! /bin/bash
#SBATCH -N 1
#SBATCH -J RFNet
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -exu

FOLD=$1
shift
HEMI=$1
shift
syn_modals=$1
shift
indiv_modals=$1
shift
lamb=$1
shift

out_modal=t
ATLAS=Glasser
chnn=180
#ATLAS=BN_Atlas
#chnn=105

msmtype=msm

#for save_dir in HCP1200 ; do
for save_dir in HCP_retest ; do
  for fold in $FOLD; do
    for hemi in $HEMI; do

      if [ $save_dir == HCP1200 ]; then
        sub_list_file=../sub_lists/sub_list_drt_fold${fold}T.txt
      elif [ $save_dir == HCP_retest ]; then
        sub_list_file=../sub_lists/sub_list_drt_fold${fold}_rt39.txt

      elif [ $save_dir == ADNI_AD ]; then
        sub_list_file=../sub_lists/sub_list_adni_ad_drt.txt
      elif [ $save_dir == ADNI_MCI ]; then
        sub_list_file=../sub_lists/sub_list_adni_mci_diag_drt.txt
      elif [ $save_dir == ADNI_NC ]; then
        sub_list_file=../sub_lists/sub_list_adni_nc_diag2_drt.txt

      elif [ $save_dir == CHCP ]; then
      sub_list_file=../sub_lists/sub_list_CHCP_dr.txt
      fi

      syn_task_name=Unet_syn_${msmtype}_${syn_modals// /}_${hemi}_task_contrast24
      indiv_prename=Unet_indiv_${msmtype}_${syn_modals// /}_lam${lamb}_${ATLAS}_${indiv_modals// /}_${hemi}

      CUDA_VISIBLE_DEVICES=$1 python3 test_indiv_sens.py \
                                      --hemi ${hemi} \
                                      --modals ${indiv_modals} \
                                      --test_sub_list_file ${sub_list_file} \
                                      --reference_atlas ${ATLAS} \
                                      --out_channels ${chnn} \
                                      --feat_dir ../features/${save_dir} \
                                      --indiv_label_dir ../indiv_atlas/${save_dir} \
                                      --lamb $lamb \
                                      --fold ${fold} \
                                      --indiv_prename $indiv_prename \
                                      --syn_task_name $syn_task_name
    done
  done
done
