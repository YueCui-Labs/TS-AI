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


from scipy import stats


def calc_matrix(sub_list, method):
    trt = 'HCP1200'
    # trt = 'HCP_retest'
    sub_num = len(sub_list)
    task_num = 24

    label_L = surface.load_surf_data(os.path.join('../atlases/fsaverage.L.Glasser.32k_fs_LR.label.gii')).astype(np.int32)
    valid_L = label_L > 0
    label_R = surface.load_surf_data(os.path.join('../atlases/fsaverage.R.Glasser.32k_fs_LR.label.gii')).astype(np.int32)
    valid_R = label_R > 0
    print(valid_L.shape, valid_R.shape)

    recon, ground = [], []
    for sub in sub_list:
        recon_L = surface.load_surf_data(
            os.path.join('../features/HCP1200', sub, '{}_L_task_contrast24.L.32k_fs_LR.func.gii'.format(method))).astype(np.float32)
        recon_L = recon_L.T if recon_L.shape[1] == 32492 else recon_L
        recon_L = recon_L[valid_L, :]

        recon_R = surface.load_surf_data(
            os.path.join('../features/HCP1200', sub, '{}_R_task_contrast24.R.32k_fs_LR.func.gii'.format(method))).astype(np.float32)
        recon_R = recon_R.T if recon_R.shape[1] == 32492 else recon_R
        recon_R = recon_R[valid_R, :]

        recon.append(np.concatenate([recon_L, recon_R], axis=0))

        ground_L = surface.load_surf_data(
            os.path.join('../features', trt, sub, 'task_contrast24_MSMAll.L.32k_fs_LR.func.gii'))[:, valid_L].T
        ground_R = surface.load_surf_data(
            os.path.join('../features', trt, sub, 'task_contrast24_MSMAll.R.32k_fs_LR.func.gii'))[:, valid_R].T
        ground.append(np.concatenate([ground_L, ground_R], axis=0))

    ## AUC
    acc_auc_mat_file = 'plot/05_{}_acc_auc.npy'.format(method)

    # if True:
    if os.path.exists(acc_auc_mat_file):
        auc_mat = np.load(acc_auc_mat_file)
    else:
        auc_mat = np.zeros((sub_num, sub_num, task_num), dtype=float)
        for i in range(sub_num):
            for j in range(sub_num):
                _, auc_mat[i, j, :] = dice_auc(recon[i], ground[j])

        np.save(acc_auc_mat_file, auc_mat)

    acc_auc = np.zeros((task_num))
    for contrast in range(task_num):
        cnt = 0
        for j in range(sub_num):
            cnt += np.argmax(auc_mat[:, j, contrast]) == j
        cnt /= sub_num
        acc_auc[contrast] = cnt
    print(acc_auc)

    ## Pearson's r
    acc_pr_mat_file = 'plot/05_{}_acc_pr.npy'.format(method)

    # if True:
    if os.path.exists(acc_pr_mat_file):
        pearsonr_mat = np.load(acc_pr_mat_file)
    else:
        pearsonr_mat = np.zeros((sub_num, sub_num, task_num), dtype=float)
        for i in range(sub_num):
            for j in range(sub_num):
                pearsonr_mat[i, j, :] = corrcoef(recon[i], ground[j])
        np.save(acc_pr_mat_file, pearsonr_mat)

    acc_pr = np.zeros((task_num))
    for contrast in range(task_num):
        cnt = 0
        # print(np.argmax(pearsonr_mat[:, :, contrast], axis=0))
        for j in range(sub_num):
            cnt += np.argmax(pearsonr_mat[:, j, contrast]) == j
        cnt /= sub_num
        acc_pr[contrast] = cnt
    print(acc_pr)

    return auc_mat, pearsonr_mat
    # return pearsonr_mat


contrast_name = [
    'Emotion - Faces', 'Emotion - Shapes',
    'Gambling - Punish', 'Gambling - Reward',
    'Language - Math', 'Language - Story',
    'Motor - Cue', 'Motor - LF', 'Motor - LH', 'Motor - RF', 'Motor - RH', 'Motor - T',
    'Relational - Match', 'Relational - Rel',
    'Social - Random', 'Social - TOM',
    'WM - 2bk body', 'WM - 2bk face', 'WM - 2bk place', 'WM - 2bk tool',
    'WM - 0bk body', 'WM - 0bk face', 'WM - 0bk place', 'WM - 0bk tool'
]

import matplotlib

matplotlib.use('Agg')

import seaborn as sns
import matplotlib.pyplot as plt


def matrix_plot(mat, measure):
    task_num = 24

    ncols = 4
    nrows = task_num // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 24))

    sns.set_theme(style="ticks", palette='pastel', font_scale=1.8)

    # Plot on the first subplot
    for i, cname in enumerate(contrast_name):
        inmat = mat[:,:,i]
        inmat = (inmat - inmat.min(0)) / (inmat.max(0) - inmat.min(0))
        sns.heatmap(inmat, cmap="hot", cbar=False,
                    # vmin=max(map(max, inmat)), vmax=min(map(min, inmat)),
                    square=True,
                    xticklabels=False, yticklabels=False,
                    ax=axes[i // ncols, i % ncols])
        axes[i // ncols, i % ncols].set_title(cname)
        axes[i // ncols, i % ncols].set(xlabel="", ylabel="")

    plt.savefig('plot/matrix_plot24_{}.tif'.format(measure))
    plt.close()


if __name__ == '__main__':
    sub_list_file, method = '../sub_lists/sub_list_drt_rt.txt', 'Unet_syn_msm_dr'

    sub_list = np.loadtxt(sub_list_file, dtype=str)
    print(sub_list.shape)

    auc_mat, pearsonr_mat = calc_matrix(sub_list, method)

    auc_mat = np.load('plot/05_{}_acc_auc.npy'.format(method))
    matrix_plot(auc_mat, 'auc')

    auc_mat = np.load('plot/05_{}_acc_pr.npy'.format(method))
    matrix_plot(auc_mat, 'pr')
