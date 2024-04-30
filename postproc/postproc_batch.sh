#! /bin/bash

#for method in \
#GMFnet_cam_std_womask_recon_gig_msm_dr-t_lam10_Glasser_drt_by-drt \
#GMFnet_cam_std_womask_recon_gig_msm_dr-t_lam10_Glasser_dr_by-dr \
#GMFnet_cam_std_womask_recon_gig_msm_dr-t_lam10_Glasser_rt_by-rt \
#GMFnet_cam_std_womask_recon_gig_msm_dr-t_lam10_Glasser_dt_by-dt \
#GMFnet_cam_std_womask_recon_gig_msm_dr-t_lam10_Glasser_r_by-r \
#GMFnet_cam_std_womask_recon_gig_msm_dr-t_lam10_Glasser_d_by-d \
#GMFnet_cam_std_womask_recon_gig_msm_dr-t_lam10_Glasser_t_by-t \
# ; do
for method in $1 ; do
#  for save_dir in HCP1200 HCP_retest; do
#  for save_dir in CHCP; do
  for save_dir in ADNI_AD ADNI_MCI ADNI_NC; do

    if [ $save_dir == HCP1200 ]; then
      sub_list_file=../sub_lists/sub_list_drt.txt
    elif [ $save_dir == HCP_retest ]; then
      sub_list_file=../sub_lists/sub_list_drt_rt.txt
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

    sbatch -o output/${method}_${save_dir}_L.out postproc.sh $sub_list_file ../indiv_atlas/${save_dir} $method L
    sbatch -o output/${method}_${save_dir}_R.out postproc.sh $sub_list_file ../indiv_atlas/${save_dir} $method R
  done
done

