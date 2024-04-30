from nilearn import surface
import numpy as np
from tqdm import tqdm
from sys import argv
import os


def dice_auc(source, target):
    """
    source, target: (V,C)
    return: (C,)
    """
    low_thld, high_thld = 5, 50

    vert_num, chnn = source.shape

    sorted_source = np.sort(source, axis=0)
    sorted_target = np.sort(target, axis=0)
    dice_curve = []

    for topk in range(low_thld, high_thld + 1):
        top_s = sorted_source[round((100 - topk) / 100 * vert_num), :]  # (C,)
        top_t = sorted_target[round((100 - topk) / 100 * vert_num), :]  # (C,)
        # print(top_s, top_t)
        sc = source > top_s[None, :]
        tc = target > top_t[None, :]
        # print((sc==tc).sum())
        # print((sc & tc).sum(0), sc.sum(0), tc.sum(0))
        dice_ = 2 * (sc & tc).sum(0) / (sc.sum(0) + tc.sum(0))
        dice_curve.append(dice_)
    dice_curve = np.stack(dice_curve, axis=0)
    # print(dice_curve)
    return dice_curve, dice_curve.sum(0) * 0.01


def corrcoef(source, target):
    """
    source, target: (V,C)
    return: (C,)
    """
    vx = source - source.mean(0, keepdims=True)
    vy = target - target.mean(0, keepdims=True)

    cost = (vx * vy).sum(0) / (np.sqrt((vx ** 2).sum(0)) * np.sqrt((vy ** 2).sum(0)) + 1e-4)
    return cost


def calc_idv_corr_drt_single(sub_list, method, dataset):
    label_L = surface.load_surf_data(os.path.join('../atlases/fsaverage.L.Glasser.32k_fs_LR.label.gii'))
    valid_L = label_L > 0
    label_R = surface.load_surf_data(os.path.join('../atlases/fsaverage.R.Glasser.32k_fs_LR.label.gii'))
    valid_R = label_R > 0
    # print(valid_L.shape, valid_R.shape)

    pearsonr, dice_curve, auc = [], [], []

    # file_name = 'result/05_{}-{}_corr_sub{}.npy'.format(method, dataset, len(sub_list))
    # if os.path.exists(file_name):
    #     pearsonr = np.load(file_name)
    #     print(pearsonr.shape)
    #     return pearsonr.flatten()

    for sub in tqdm(sub_list):
        recon_L = surface.load_surf_data(
            os.path.join('../features', dataset, sub, '{}_L_task_contrast24.L.32k_fs_LR.func.gii'.format(method))
        ).astype(np.float32)
        recon_L = recon_L.T if recon_L.shape[1] == 32492 else recon_L
        recon_L = recon_L[valid_L, :]

        recon_R = surface.load_surf_data(
            os.path.join('../features', dataset, sub, '{}_R_task_contrast24.R.32k_fs_LR.func.gii'.format(method))
        ).astype(np.float32)
        recon_R = recon_R.T if recon_R.shape[1] == 32492 else recon_R
        recon_R = recon_R[valid_R, :]

        recon = np.concatenate([recon_L, recon_R], axis=0)

        ground_L = surface.load_surf_data(
            os.path.join('../features', dataset, sub, 'task_contrast24_MSMAll.L.32k_fs_LR.func.gii')
        ).astype(np.float32)
        ground_L = ground_L.T if ground_L.shape[1] == 32492 else ground_L
        ground_L = ground_L[valid_L, :]

        ground_R = surface.load_surf_data(
            os.path.join('../features', dataset, sub, 'task_contrast24_MSMAll.R.32k_fs_LR.func.gii')
        ).astype(np.float32)
        ground_R = ground_R.T if ground_R.shape[1] == 32492 else ground_R
        ground_R = ground_R[valid_R, :]

        ground = np.concatenate([ground_L, ground_R], axis=0)
        # print(recon.shape, ground.shape)

        dice_curve_, auc_ = dice_auc(recon, ground)
        dice_curve.append(dice_curve_)
        auc.append(auc_)

        pearsonr.append(corrcoef(recon, ground))

    dice_curve = np.stack(dice_curve, axis=0)
    auc = np.stack(auc, axis=0)
    pearsonr = np.stack(pearsonr, axis=0)

    print('{:.3f} ± {:.3f}\t{:.3f} ± {:.3f}'.format(
        pearsonr.mean(), pearsonr.std(),
        auc.mean(), auc.std(),
    ))
    np.save('result/05_task_syn_{}_{}.npy'.format(method, dataset), {'pearsonr':pearsonr, 'dice_curve':dice_curve, 'auc':auc})


def calc_grp_corr(sub_list, dataset):
    label_L = surface.load_surf_data(os.path.join('../atlases/fsaverage.L.Glasser.32k_fs_LR.label.gii'))
    valid_L = label_L > 0
    label_R = surface.load_surf_data(os.path.join('../atlases/fsaverage.R.Glasser.32k_fs_LR.label.gii'))
    valid_R = label_R > 0
    print(valid_L.shape, valid_R.shape)

    recon_L = surface.load_surf_data('../features/avg_task_contrast24_MSMAll.{}.L.32k_fs_LR.func.gii'.format('HCP1200'))
    recon_L = recon_L.T if recon_L.shape[1] == 32492 else recon_L
    recon_L = recon_L[valid_L, :]

    recon_R = surface.load_surf_data('../features/avg_task_contrast24_MSMAll.{}.R.32k_fs_LR.func.gii'.format('HCP1200'))
    recon_R = recon_R.T if recon_R.shape[1] == 32492 else recon_R
    recon_R = recon_R[valid_R, :]

    recon = np.concatenate([recon_L, recon_R], axis=0)

    pearsonr, dice_curve, auc = [], [], []

    for sub in tqdm(sub_list):
        ground_L = surface.load_surf_data(
            os.path.join('../features', dataset, sub, 'task_contrast24_MSMAll.L.32k_fs_LR.func.gii')
        ).astype(np.float32)
        ground_L = ground_L.T if ground_L.shape[1] == 32492 else ground_L
        ground_L = ground_L[valid_L, :]

        ground_R = surface.load_surf_data(
            os.path.join('../features', dataset, sub, 'task_contrast24_MSMAll.R.32k_fs_LR.func.gii')
        ).astype(np.float32)
        ground_R = ground_R.T if ground_R.shape[1] == 32492 else ground_R
        ground_R = ground_R[valid_R, :]

        ground = np.concatenate([ground_L, ground_R], axis=0)
        # print(recon.shape, ground.shape)

        dice_curve_, auc_ = dice_auc(recon, ground)
        dice_curve.append(dice_curve_)
        auc.append(auc_)

        pearsonr.append(corrcoef(recon, ground))

    dice_curve = np.stack(dice_curve, axis=0)
    auc = np.stack(auc, axis=0)

    pearsonr = np.stack(pearsonr, axis=0)

    print('{:.3f} ± {:.3f}\t{:.3f} ± {:.3f}'.format(
        pearsonr.mean(), pearsonr.std(),
        auc.mean(), auc.std(),
    ))
    np.save('result/05_task_syn_group_{}.npy'.format(dataset), {'pearsonr':pearsonr, 'dice_curve':dice_curve, 'auc':auc})


def star(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    else:
        return "n.s."


from scipy import stats
from statsmodels.stats.multitest import multipletests


def response_main():
    # dataset = 'HCP1200'
    dataset = 'HCP_retest'

    sub_list = np.loadtxt('../sub_lists/sub_list_drt_rt.txt', dtype=str)
    print(sub_list.shape)

    # calc_idv_corr(sub_list)
    # calc_grp_corr(sub_list)
    s2_drg = calc_idv_corr_drt_single(sub_list, 'Unet_syn_msm_drg', dataset)
    s2_dr = calc_idv_corr_drt_single(sub_list, 'Unet_syn_msm_dr', dataset)
    b2_drg = calc_idv_corr_drt_single(sub_list, 'Unet_2branch_msm_outfeat_lamf10_lams10_Glasser_drg', dataset)
    b2_dr = calc_idv_corr_drt_single(sub_list, 'Unet_2branch_msm_outfeat_lamf10_lams10_Glasser_dr', dataset)

    _, p1 = stats.ttest_rel(s2_dr, s2_drg)
    _, p2 = stats.ttest_rel(s2_dr, b2_drg)
    _, p3 = stats.ttest_rel(s2_dr, b2_dr)

    qvals = multipletests([p1, p2, p3], method='fdr_bh')[1]
    print(star(p1), star(p2), star(p3))
    print(star(qvals[0]), star(qvals[1]), star(qvals[2]))


def main():
    sub_list = np.loadtxt('../sub_lists/sub_list_drt_rt.txt', dtype=str)
    # calc_idv_corr_drt_single(sub_list, 'Unet_syn_msm_dr', 'HCP1200')
    # calc_idv_corr_drt_single(sub_list, 'Unet_syn_msm_dr', 'HCP_retest')

    calc_grp_corr(sub_list, 'HCP1200')
    calc_grp_corr(sub_list, 'HCP_retest')


if __name__ == '__main__':
    main()