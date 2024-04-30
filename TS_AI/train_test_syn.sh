#! /bin/bash
#SBATCH -N 1
#SBATCH -J RFNet
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -exu

FOLD=$1
shift
HEMI=$1
shift
modals=$1
shift
out_modal=$1
shift
ATLAS=Glasser
chnn=180
#ATLAS=BN_Atlas
#chnn=105

for fold in $FOLD; do
  for hemi in $HEMI; do

    syn_prename=Unet_syn_msm_${modals// /}_${hemi}
    echo $syn_prename

    CUDA_VISIBLE_DEVICES=$1 python train_syn.py \
                                    --modals ${modals} \
                                    --out_modal ${out_modal} \
                                    --hemi $hemi \
                                    --reference_atlas ${ATLAS} \
                                    --out_channels ${chnn} \
                                    --train_sub_list_file "../sub_lists/sub_list_drt_fold${fold}P.txt" \
                                    --vali_sub_list_file "../sub_lists/sub_list_drt_fold${fold}V.txt" \
                                    --batch_size 8 \
                                    --num_workers 8 \
                                    --epoch_num 30 \
                                    --fold $fold \
                                    --syn_prename $syn_prename \
                                    --feat_dir ../features/HCP1200

    # for deriving train/val syn. task

    syn_task_name=${syn_prename}_trainval_task_contrast24
    save_dir=HCP1200
    for sublist in "../sub_lists/sub_list_drt_fold${fold}P.txt" "../sub_lists/sub_list_drt_fold${fold}V.txt"; do
      CUDA_VISIBLE_DEVICES=$1 python3 test_syn.py \
                                      --hemi ${hemi} \
                                      --modals ${modals} \
                                      --out_modal ${out_modal} \
                                      --test_sub_list_file $sublist \
                                      --reference_atlas ${ATLAS} \
                                      --out_channels ${chnn} \
                                      --feat_dir ../features/${save_dir} \
                                      --indiv_label_dir ../indiv_atlas/${save_dir} \
                                      --syn_prename $syn_prename \
                                      --syn_task_name $syn_task_name \
                                      --fold ${fold}
    done
  done
done

#for save_dir in CHCP; do
for save_dir in HCP1200 HCP_retest; do
#for save_dir in ADNI_AD ADNI_MCI ADNI_NC; do
  for fold in $FOLD; do
    for hemi in $HEMI; do

      if [ $save_dir == HCP1200 ]; then
        sub_list_file=../sub_lists/sub_list_drt_fold${fold}T.txt
      elif [ $save_dir == HCP_retest ]; then
        sub_list_file=../sub_lists/sub_list_drt_fold${fold}_rt39.txt
      elif [ $save_dir == UKB_MDD ]; then
        sub_list_file=../sub_lists/sub_list_ukb_mdd_578.txt
      elif [ $save_dir == UKB_NC ]; then
        sub_list_file=../sub_lists/sub_list_ukb_nc_580.txt

        elif [ $save_dir == ADNI_AD ]; then
          sub_list_file=../sub_lists/sub_list_adni_ad_drt.txt
        elif [ $save_dir == ADNI_MCI ]; then
          sub_list_file=../sub_lists/sub_list_adni_mci_diag_drt.txt
        elif [ $save_dir == ADNI_NC ]; then
          sub_list_file=../sub_lists/sub_list_adni_nc_diag2_drt.txt

      elif [ $save_dir == CHCP ]; then
        sub_list_file=../sub_lists/sub_list_CHCP_dr.txt
      fi

      syn_prename=Unet_syn_msm_${modals// /}_${hemi}
      syn_task_name=${syn_prename}_task_contrast24

      # ../sub_lists/sub_list_drt.txt
      CUDA_VISIBLE_DEVICES=$1 python3 test_syn.py \
                                      --hemi ${hemi} \
                                      --modals ${modals} \
                                      --out_modal ${out_modal} \
                                      --test_sub_list_file ${sub_list_file} \
                                      --reference_atlas ${ATLAS} \
                                      --out_channels ${chnn} \
                                      --feat_dir ../features/${save_dir} \
                                      --indiv_label_dir ../indiv_atlas/${save_dir} \
                                      --syn_prename $syn_prename \
                                      --syn_task_name $syn_task_name \
                                      --fold ${fold}
    done
  done
done
