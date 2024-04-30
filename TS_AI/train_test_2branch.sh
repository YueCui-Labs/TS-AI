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
modals=$1
shift
lamb=$1
shift
lamb_syn=$1
shift

out_modal=t
ATLAS=Glasser
chnn=180
#ATLAS=BN_Atlas
#chnn=105

msmtype=msm_outfeat

#for fold in $FOLD; do
#  for hemi in $HEMI; do
#
#    prename=Unet_2branch_${msmtype}_lamf${lamb}_lams${lamb_syn}_${ATLAS}_${modals// /}_${hemi}
#
#    CUDA_VISIBLE_DEVICES=$1 python3 train_2branch.py \
#                                    --modals ${modals} \
#                                    --out_modal ${out_modal} \
#                                    --hemi $hemi \
#                                    --reference_atlas ${ATLAS} \
#                                    --out_channels ${chnn} \
#                                    --train_sub_list_file "../sub_lists/sub_list_drt_fold${fold}P.txt" \
#                                    --vali_sub_list_file "../sub_lists/sub_list_drt_fold${fold}V.txt" \
#                                    --batch_size 4 \
#                                    --num_workers 4 \
#                                    --epoch_num 30 \
#                                    --fold $fold \
#                                    --lamb $lamb \
#                                    --lamb_syn $lamb_syn \
#                                    --syn_prename $prename \
#                                    --feat_dir ../features/HCP1200
#  done
#done

#for save_dir in CHCP; do
for save_dir in HCP1200 HCP_retest ; do
#for save_dir in ADNI_AD ADNI_MCI ADNI_NC; do
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

      prename=Unet_2branch_${msmtype}_lamf${lamb}_lams${lamb_syn}_${ATLAS}_${modals// /}_${hemi}

      CUDA_VISIBLE_DEVICES=$1 python3 test_2branch.py \
                                      --hemi ${hemi} \
                                      --modals ${modals} \
                                      --out_modal ${out_modal} \
                                      --test_sub_list_file ${sub_list_file} \
                                      --reference_atlas ${ATLAS} \
                                      --out_channels ${chnn} \
                                      --feat_dir ../features/${save_dir} \
                                      --indiv_label_dir ../indiv_atlas/${save_dir} \
                                      --lamb $lamb \
                                      --lamb_syn $lamb_syn \
                                      --syn_prename $prename \
                                      --fold ${fold}
    done
  done
done
