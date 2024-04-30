import numpy as np
import statsmodels.api as sm
import scipy.special
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
from pygam import LinearGAM, s, l, f
from scipy.stats import norm, pearsonr
import pandas as pd


import scipy.io as scio
from nilearn import surface
import os, subprocess

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

def glm_group_size(method):

    ref_L = surface.load_surf_data(
        '../atlases/fsaverage.L.Glasser.32k_fs_LR.label.gii'
    ).astype(np.float32)

    ref_R = surface.load_surf_data(
        '../atlases/fsaverage.R.Glasser.32k_fs_LR.label.gii'
    ).astype(np.float32)

    sex_list = np.loadtxt('adni/NC_MCI_AD_sex.txt', dtype=str)
    age_list = np.loadtxt('adni/NC_MCI_AD_age.txt', dtype=str)
    rms_list = np.loadtxt('adni/NC_MCI_AD_relRMS.txt', dtype=str)
    etiv_list = np.loadtxt('adni/NC_MCI_AD_eTIV.txt', dtype=str)

    dicsex, dicage, dicrms, dicetiv = {}, {}, {}, {}
    for ss in sex_list:
        dicsex[ss[0]] = ss[1]
    for ss in age_list:
        dicage[ss[0]] = ss[1]
    for ss in rms_list:
        dicrms[ss[0]] = ss[1]
    for ss in etiv_list:
        dicetiv[ss[0]] = float(ss[1])

    sub_lists = [
        ('../sub_lists/sub_list_adni_nc_diag2_drt.txt', 'NC'),
        ('../sub_lists/sub_list_adni_mci_diag_drt.txt', 'MCI'),
        ('../sub_lists/sub_list_adni_ad_drt.txt', 'AD')
    ]

    topo_all_L, topo_all_R = [], []
    sex, age, rms, etiv, y = [], [], [], [], []

    surf_dir = '/data0/share/projects/Cui_Group_Files/ADNI/data'

    for i, (sub_list_name, brev) in enumerate(sub_lists):
        sub_list = np.loadtxt(sub_list_name, dtype=str)

        for sub in sub_list:
            # left
            surf_area_L = "{}/{}/{}/T1w/fsaverage_LR32k/{}.L.midthickness_area.32k_fs_LR.surf.gii".format(surf_dir, brev, sub, sub)
            if not os.path.exists(surf_area_L):
                surf_L = "{}/{}/{}/T1w/fsaverage_LR32k/{}.L.midthickness.32k_fs_LR.surf.gii".format(surf_dir, brev, sub, sub)
                p = subprocess.Popen("wb_command -surface-vertex-areas {} {}".format(
                    surf_L, surf_area_L),
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                return_code = p.wait()
                if return_code != 0:
                    print('wb_command -surface-vertex-areas failed')
                    exit(1)

            surf_area_L = surface.load_surf_data(surf_area_L).astype(np.float32)  #

            topo_subj_L = []
            for parc in range(181, 361):
                topo_subj_L.append(surf_area_L[ref_L == parc].sum() / np.sum(ref_L == parc))
            topo_all_L.append(np.array(topo_subj_L) / surf_area_L[ref_L >0].sum())

            # right
            surf_area_R = "{}/{}/{}/T1w/fsaverage_LR32k/{}.R.midthickness_area.32k_fs_LR.surf.gii".format(surf_dir, brev, sub, sub)
            if not os.path.exists(surf_area_R):
                surf_R = "{}/{}/{}/T1w/fsaverage_LR32k/{}.R.midthickness.32k_fs_LR.surf.gii".format(surf_dir, brev, sub, sub)
                p = subprocess.Popen("wb_command -surface-vertex-areas {} {}".format(
                    surf_R, surf_area_R),
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                return_code = p.wait()
                if return_code != 0:
                    print('wb_command -surface-vertex-areas failed')
                    exit(1)

            surf_area_R = surface.load_surf_data(surf_area_L).astype(np.float32)  #

            topo_subj_R = []
            for parc in range(1, 181):
                topo_subj_R.append(surf_area_R[ref_R == parc].sum() / np.sum(ref_R == parc))
            topo_all_R.append(np.array(topo_subj_R) / surf_area_R[ref_R >0].sum())

            # scalar
            sex.append(dicsex[sub])
            age.append(dicage[sub])
            rms.append(dicrms[sub])
            etiv.append(dicetiv[sub])
            y.append(i)

    topo_all_L = np.stack(topo_all_L, axis=0)
    topo_all_R = np.stack(topo_all_R, axis=0)
    print(topo_all_L.shape, topo_all_R.shape)

    topo = np.concatenate([topo_all_R, topo_all_L], axis=1)  # Subj x Parc
    print(topo.size)

    # np.save('adni/parc_loading_{}.{}.npy'.format(
    np.save('adni/parc_norm_loading_{}.{}.npy'.format(
        method, '_'.join([kk[1] for kk in sub_lists])
    ), topo)

    pv = []
    tv = []
    for parc in range(360):
        X = pd.DataFrame({
            "topo": topo[:, parc],
            "sex": sex,
            "age": age,
            "rms": rms,
            "etiv": etiv
        })
        X = pd.get_dummies(X, columns=['sex'], drop_first=True)
        X = sm.add_constant(X)
        # print(X)

        model = sm.GLM(y, X.astype(float),
                       family=sm.families.Gaussian()
                       # family=sm.families.Binomial()
                       # family=sm.families.Binomial(link=sm.genmod.families.links.logit())
                       )
        result = model.fit()

        pv.append(result.pvalues[1])
        tv.append(result.tvalues[1])

    pv = multipletests(pv, method='fdr_bh')[1]
    print((pv < 0.05).sum(), np.where(pv < 0.05))

    np.save('adni/parc_loading_tval_{}.{}.npy'.format(method, '_'.join([kk[1] for kk in sub_lists])),
            {"pv": pv, "tv": tv})

    adv_L = np.zeros_like(ref_L, dtype=np.float32)
    adv_R = np.zeros_like(ref_R, dtype=np.float32)
    print(adv_L.shape, adv_R.shape)
    for i in range(360):
        adv_L[ref_L == (i + 1)] = tv[i]
        adv_R[ref_R == (i + 1)] = tv[i]

    import nibabel as nib
    template = nib.load('../atlases/fsaverage.L.Glasser.32k_fs_LR.label.gii')
    template.remove_gifti_data_array(0)
    template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(adv_L))
    nib.loadsave.save(template, 'adni/parc_loading_tval_{}.{}.L.32k_fs_LR.func.gii'.format(method, '_'.join(
        [kk[1] for kk in sub_lists])))

    template = nib.load('../atlases/fsaverage.R.Glasser.32k_fs_LR.label.gii')
    template.remove_gifti_data_array(0)
    template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(adv_R))
    nib.loadsave.save(template, 'adni/parc_loading_tval_{}.{}.R.32k_fs_LR.func.gii'.format(method, '_'.join(
        [kk[1] for kk in sub_lists])))

    pt_L = np.zeros_like(ref_L, dtype=np.float32)
    pt_R = np.zeros_like(ref_R, dtype=np.float32)
    print(pt_L.shape, pt_R.shape)
    for i in range(360):
        if pv[i] >= 0.05:
            continue
        pt_L[ref_L == (i + 1)] = tv[i]
        pt_R[ref_R == (i + 1)] = tv[i]

    template = nib.load('../atlases/fsaverage.L.Glasser.32k_fs_LR.label.gii')
    template.remove_gifti_data_array(0)
    template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pt_L))
    nib.loadsave.save(template, 'adni/parc_loading_pthld_tval_{}.{}.L.32k_fs_LR.func.gii'.format(method, '_'.join(
        [kk[1] for kk in sub_lists])))

    template = nib.load('../atlases/fsaverage.R.Glasser.32k_fs_LR.label.gii')
    template.remove_gifti_data_array(0)
    template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pt_R))
    nib.loadsave.save(template, 'adni/parc_loading_pthld_tval_{}.{}.R.32k_fs_LR.func.gii'.format(method, '_'.join(
        [kk[1] for kk in sub_lists])))


def glm_size_neigh(method):
    ref_L = surface.load_surf_data(
        '../atlases/fsaverage.L.Glasser.32k_fs_LR.label.gii'
    ).astype(np.float32)
    atlas_mask_L = ref_L > 0

    ref_R = surface.load_surf_data(
        '../atlases/fsaverage.R.Glasser.32k_fs_LR.label.gii'
    ).astype(np.float32)
    atlas_mask_R = ref_R > 0

    neigh_L = surface.load_surf_data(
        '../atlases/fsaverage.L.Glasser_search_light_r20mm.32k_fs_LR.func.gii'
    ).astype(bool)
    neigh_L = neigh_L[atlas_mask_L,:]

    neigh_R = surface.load_surf_data(
        '../atlases/fsaverage.R.Glasser_search_light_r20mm.32k_fs_LR.func.gii'
    ).astype(bool)
    neigh_R = neigh_R[atlas_mask_R,:]

    print(neigh_L.shape, neigh_R.shape)

    sex_list = np.loadtxt('adni/NC_MCI_AD_sex.txt', dtype=str)
    age_list = np.loadtxt('adni/NC_MCI_AD_age.txt', dtype=str)
    rms_list = np.loadtxt('adni/NC_MCI_AD_relRMS.txt', dtype=str)
    etiv_list = np.loadtxt('adni/NC_MCI_AD_eTIV.txt', dtype=str)

    dicsex, dicage, dicrms, dicetiv = {}, {}, {}, {}
    for ss in sex_list:
        dicsex[ss[0]] = ss[1]
    for ss in age_list:
        dicage[ss[0]] = ss[1]
    for ss in rms_list:
        dicrms[ss[0]] = ss[1]
    for ss in etiv_list:
        dicetiv[ss[0]] = float(ss[1])

    sub_lists = [
        ('../sub_lists/sub_list_adni_nc_diag2_drt.txt', 'NC'),
        ('../sub_lists/sub_list_adni_mci_diag_drt.txt', 'MCI'),
        ('../sub_lists/sub_list_adni_ad_drt.txt', 'AD')
    ]

    topo_all_L, topo_all_R = [], []
    sex, age, rms, etiv, y = [], [], [], [], []

    for i, (sub_list_name, brev) in enumerate(sub_lists):
        sub_list = np.loadtxt(sub_list_name, dtype=str)

        topo_L = []
        for sub in sub_list:
            label_L = surface.load_surf_data(
                '../indiv_atlas/ADNI_{}/{}/{}_L_prob.32k_fs_LR.func.gii'.format(brev, sub, method)
            ).astype(np.float32)
            topo_L.append(label_L[atlas_mask_L > 0, :])
        topo_L = np.stack(topo_L, axis=0)  # subj x #vert x #parc

        topo_R = []
        for sub in sub_list:
            label_R = surface.load_surf_data(
                '../indiv_atlas/ADNI_{}/{}/{}_R_prob.32k_fs_LR.func.gii'.format(brev, sub, method)
            ).astype(np.float32)
            topo_R.append(label_R[atlas_mask_R > 0, :])
        topo_R = np.stack(topo_R, axis=0)  # subj x #vert x #parc

        topo_all_L.append(topo_L)
        topo_all_R.append(topo_R)

        for sub in sub_list:
            sex.append(dicsex[sub])
            age.append(dicage[sub])
            rms.append(dicrms[sub])
            etiv.append(dicetiv[sub])
            y.append(i)

    topo_all_L = np.concatenate(topo_all_L, axis=0)
    topo_all_R = np.concatenate(topo_all_R, axis=0)
    print(topo_all_L.shape, topo_all_R.shape)

    # topo_all_L[topo_all_L<0] = 0
    # topo_all_R[topo_all_R<0] = 0

    # topo_all_L = scipy.special.softmax(topo_all_L, axis=2)
    # topo_all_R = scipy.special.softmax(topo_all_R, axis=2)
    topo_all_L[:, ~neigh_L] = 0
    topo_all_R[:, ~neigh_R] = 0
    topo = np.concatenate([topo_all_R.sum(1), topo_all_L.sum(1)], axis=1)  # Subj x Parc
    ref_size_L = []
    for parc in range(181,361):
        ref_size_L.append(np.sum(ref_L == parc))
    ref_size_L = np.array(ref_size_L)

    ref_size_R = []
    for parc in range(1,181):
        ref_size_R.append(np.sum(ref_R == parc))
    ref_size_R = np.array(ref_size_R)
    topo = topo / np.concatenate([ref_size_R, ref_size_L])[None, :]
    print(topo.size)

    # np.save('adni/parc_loading_{}.{}.npy'.format(
    np.save('adni/parc_norm_loading_{}.{}.npy'.format(
        method, '_'.join([kk[1] for kk in sub_lists])
    ), topo)

    # print(topo)
    # print(topo.shape, y.shape)

    pv = []
    tv = []
    for parc in range(360):
        X = pd.DataFrame({
            "topo": topo[:, parc],
            "sex": sex,
            "age": age,
            "rms": rms,
            "etiv": etiv
        })
        X = pd.get_dummies(X, columns=['sex'], drop_first=True)
        X = sm.add_constant(X)
        # print(X)

        model = sm.GLM(y, X.astype(float),
                       family=sm.families.Gaussian()
                       # family=sm.families.Binomial()
                       # family=sm.families.Binomial(link=sm.genmod.families.links.logit())
                       )
        result = model.fit()

        pv.append(result.pvalues[1])
        tv.append(result.tvalues[1])

    pv = multipletests(pv, method='fdr_bh')[1]
    print((pv < 0.05).sum(), np.where(pv < 0.05))

    np.save('adni/parc_loading_tval_{}.{}.npy'.format(method, '_'.join([kk[1] for kk in sub_lists])),
            {"pv": pv, "tv": tv})

    adv_L = np.zeros_like(ref_L, dtype=np.float32)
    adv_R = np.zeros_like(ref_R, dtype=np.float32)
    print(adv_L.shape, adv_R.shape)
    for i in range(360):
        adv_L[ref_L == (i + 1)] = tv[i]
        adv_R[ref_R == (i + 1)] = tv[i]

    import nibabel as nib
    template = nib.load('../atlases/fsaverage.L.Glasser.32k_fs_LR.label.gii')
    template.remove_gifti_data_array(0)
    template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(adv_L))
    nib.loadsave.save(template, 'adni/parc_loading_tval_{}.{}.L.32k_fs_LR.func.gii'.format(method, '_'.join(
        [kk[1] for kk in sub_lists])))

    template = nib.load('../atlases/fsaverage.R.Glasser.32k_fs_LR.label.gii')
    template.remove_gifti_data_array(0)
    template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(adv_R))
    nib.loadsave.save(template, 'adni/parc_loading_tval_{}.{}.R.32k_fs_LR.func.gii'.format(method, '_'.join(
        [kk[1] for kk in sub_lists])))

    pt_L = np.zeros_like(ref_L, dtype=np.float32)
    pt_R = np.zeros_like(ref_R, dtype=np.float32)
    print(pt_L.shape, pt_R.shape)
    for i in range(360):
        if pv[i] >= 0.05:
            continue
        pt_L[ref_L == (i + 1)] = tv[i]
        pt_R[ref_R == (i + 1)] = tv[i]

    template = nib.load('../atlases/fsaverage.L.Glasser.32k_fs_LR.label.gii')
    template.remove_gifti_data_array(0)
    template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pt_L))
    nib.loadsave.save(template, 'adni/parc_loading_pthld_tval_{}.{}.L.32k_fs_LR.func.gii'.format(method, '_'.join(
        [kk[1] for kk in sub_lists])))

    template = nib.load('../atlases/fsaverage.R.Glasser.32k_fs_LR.label.gii')
    template.remove_gifti_data_array(0)
    template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pt_R))
    nib.loadsave.save(template, 'adni/parc_loading_pthld_tval_{}.{}.R.32k_fs_LR.func.gii'.format(method, '_'.join(
        [kk[1] for kk in sub_lists])))


def corr_loading_parc_size(method):
    # method = 'Unet_indiv_msm_dr_lam10_Glasser_drt_fold1'
    indiv_name = 'Unet_indiv_msm_dr_lam10_Glasser_drt_post'

    ref_L = surface.load_surf_data(
        '../atlases/fsaverage.L.Glasser.32k_fs_LR.label.gii'
    ).astype(np.int32)

    ref_R = surface.load_surf_data(
        '../atlases/fsaverage.R.Glasser.32k_fs_LR.label.gii'
    ).astype(np.int32)

    ref_size_L = []
    for parc in range(181,361):
        ref_size_L.append(np.sum(ref_L == parc))
    ref_size_L = np.array(ref_size_L)

    ref_size_R = []
    for parc in range(1,181):
        ref_size_R.append(np.sum(ref_R == parc))
    ref_size_R = np.array(ref_size_R)

    ref_size = np.concatenate([ref_size_R, ref_size_L])

    sub_lists = [
        ('../sub_lists/sub_list_adni_nc_diag2_drt.txt', 'NC'),
        ('../sub_lists/sub_list_adni_mci_diag_drt.txt', 'MCI'),
        ('../sub_lists/sub_list_adni_ad_drt.txt', 'AD')
    ]

    nc_num = np.loadtxt(sub_lists[0][0], dtype=str).size

    norm_loading = np.load('adni/parc_loading_{}.{}.npy'.format(
        method, '_'.join([kk[1] for kk in sub_lists])
    ))

    loading = norm_loading * ref_size[None, :]

    from scipy.stats import linregress
    slope, intercept, _, _, _ = linregress(ref_size, loading[:nc_num,:].mean(0))
    norm_loading = loading - (slope * ref_size[None, :] + intercept)

    cnt = 0
    for kk in sub_lists:
        sub_num = np.loadtxt(kk[0], dtype=str).size
        np.save('adni/parc_norm_loading_{}.{}.npy'.format(
        method, kk[1]), norm_loading[cnt:cnt+sub_num,:])
        cnt += sub_num

    np.save('adni/parc_norm_loading_{}.{}.npy'.format(
        method, '_'.join([kk[1] for kk in sub_lists])
    ), norm_loading)

    r, pv = pearsonr(norm_loading.mean(0), ref_size)
    print('norm loading: {:.2f}, {:.2e}'.format(r, pv))

    r, pv = pearsonr(loading.mean(0), ref_size)
    print('loading: {:.2f}, {:.2e}'.format(r, pv))

    # parc_size = []
    # for i, (sub_list_name, brev) in enumerate(sub_lists):
    #     sub_list = np.loadtxt(sub_list_name, dtype=str)
    #
    #     for sub in sub_list:
    #         parc_indiv = []
    #         label_L = surface.load_surf_data(
    #             '../indiv_atlas/ADNI_{}/{}/{}_L.32k_fs_LR.label.gii'.format(brev, sub, indiv_name)
    #         ).astype(np.int32)
    #
    #         label_R = surface.load_surf_data(
    #             '../indiv_atlas/ADNI_{}/{}/{}_R.32k_fs_LR.label.gii'.format(brev, sub, indiv_name)
    #         ).astype(np.int32)
    #
    #         for parc in range(1,181):
    #             parc_indiv.append(np.sum(label_R == parc))
    #
    #         for parc in range(181,361):
    #             parc_indiv.append(np.sum(label_L == parc))
    #         parc_size.append(np.array(parc_indiv))
    #
    # parc_size = np.stack(parc_size, axis=0)
    # print(parc_size.shape)

    # r, pv = pearsonr(norm_loading.flatten(), parc_size.flatten())
    # print('norm loading: ', r, pv)
    #
    # r, pv = pearsonr(loading.flatten(), parc_size.flatten())
    # print('loading: ', r, pv)

    df = {
        "loading": norm_loading.mean(0),
        "ref_size": ref_size
    }
    df = pd.DataFrame(df)
    # print(df)
    sns.set_theme(style="ticks", font_scale=1.8)
    grid = sns.lmplot(data=df,
                      x="loading", y="ref_size",
                      palette='Set1', legend=False, fit_reg=True,
                      scatter_kws={"s": 10}
                      )

    grid.set(xlabel=None, ylabel=None, xticks=[])

    plt.tight_layout()
    plt.savefig('adni/01_corr_parc_size_norm_loading_{}.tif'.format(method))
    plt.close()


    df = {
        "loading": loading.mean(0),
        "ref_size": ref_size
    }
    df = pd.DataFrame(df)
    # print(df)
    sns.set_theme(style="ticks", font_scale=1.8)
    grid = sns.lmplot(data=df,
                      x="loading", y="ref_size",
                      palette='Set1', legend=False, fit_reg=True,
                      scatter_kws={"s": 10}
                      )

    grid.set(xlabel=None, ylabel=None, xticks=[])

    plt.tight_layout()
    plt.savefig('adni/01_corr_parc_size_loading_{}.tif'.format(method))
    plt.close()


def corr_size_score(method, ispos, group=['NC','MCI','AD']):
    ref_L = surface.load_surf_data(
        '../atlases/fsaverage.L.Glasser.32k_fs_LR.label.gii'
    ).astype(np.float32)
    atlas_mask_L = ref_L > 0

    ref_R = surface.load_surf_data(
        '../atlases/fsaverage.R.Glasser.32k_fs_LR.label.gii'
    ).astype(np.float32)
    atlas_mask_R = ref_R > 0

    neigh_L = surface.load_surf_data(
        '../atlases/fsaverage.L.Glasser_search_light_r20mm.32k_fs_LR.func.gii'
    ).astype(bool)
    neigh_L = neigh_L[atlas_mask_L,:]

    neigh_R = surface.load_surf_data(
        '../atlases/fsaverage.R.Glasser_search_light_r20mm.32k_fs_LR.func.gii'
    ).astype(bool)
    neigh_R = neigh_R[atlas_mask_R,:]

    print(neigh_L.shape, neigh_R.shape)

    sex_list = np.loadtxt('adni/NC_MCI_AD_sex.txt', dtype=str)
    age_list = np.loadtxt('adni/NC_MCI_AD_age.txt', dtype=str)
    rms_list = np.loadtxt('adni/NC_MCI_AD_relRMS.txt', dtype=str)
    etiv_list = np.loadtxt('adni/NC_MCI_AD_eTIV.txt', dtype=str)
    dicsex, dicage, dicrms, dicetiv = {}, {}, {}, {}
    for ss in sex_list:
        dicsex[ss[0]] = ss[1]
    for ss in age_list:
        dicage[ss[0]] = int(ss[1])
    for ss in rms_list:
        dicrms[ss[0]] = float(ss[1])
    for ss in etiv_list:
        dicetiv[ss[0]] = float(ss[1])

    sub_lists = []
    if 'NC' in group:
        sub_lists.append(('../sub_lists/sub_list_adni_nc_diag2_drt.txt', 'NC'))
    if 'MCI' in group:
        sub_lists.append(('../sub_lists/sub_list_adni_mci_diag_drt.txt', 'MCI'))
    if 'AD' in group:
        sub_lists.append(('../sub_lists/sub_list_adni_ad_drt.txt', 'AD'))

    # topo = np.load('adni/parc_loading_{}.{}.npy'.format(
    topo = []
    for kk in sub_lists:
        topo.append(np.load('adni/parc_norm_loading_{}.{}.npy'.format(
        method, kk[1]))
        )
    topo = np.concatenate(topo, axis=0)

    tmp = np.load('adni/parc_loading_tval_{}.{}.npy'.format(method, 'NC_MCI_AD'),
                  allow_pickle=True).item()

    if ispos:
        pv_ind = (np.array(tmp['pv']) < 0.05) * (np.array(tmp['tv']) > 0)
    else:
        pv_ind = (np.array(tmp['pv']) < 0.05) * (np.array(tmp['tv']) < 0)

    topo = topo[:, pv_ind].sum(1)

    # score_names = [
    #     'ADA13', 'CDRSB', 'FAQ', 'MMSE', 'MOCA'
    # ]
    score_names = [
        'ADA13', 'FAQ', 'MOCA'
    ]

    for score_name in score_names:
        score_list = np.loadtxt('adni/ADNI_NC_MCI_AD_{}.txt'.format(score_name),
                                dtype=str)

        dicscore = {}
        for ss in score_list:
            if ss[1] != 'nan':
                dicscore[ss[0]] = float(ss[1])

        sex, age, rms, etiv, y = [], [], [], [], []
        topo_vali = []
        type = []
        cnt = 0
        for i, (sub_list_name, brev) in enumerate(sub_lists):
            sub_list = np.loadtxt(sub_list_name, dtype=str)
            for sub in sub_list:
                if sub in dicscore.keys():
                    topo_vali.append(topo[cnt])

                    sex.append(dicsex[sub])
                    age.append(dicage[sub])
                    rms.append(dicrms[sub])
                    etiv.append(dicetiv[sub])

                    type.append(i)
                    y.append(dicscore[sub])
                cnt += 1
        ##

        r, pv = pearsonr(np.array(topo_vali), np.array(y))
        np.save('adni/regress_{}_{}_{}_{}.npy'.format('pos' if ispos else 'neg', method, '_'.join([kk[1] for kk in sub_lists]), score_name),
                {'topo_vali': topo_vali,
                 'y': y, 'sex':sex, 'age':age, 'rms': rms, 'etiv': etiv,
                 'type': type, 'r': r, 'pv': pv}
                )

        print('r = {:.2f}, p = {:.1e}'.format(r, pv))


def bar_plot_3groups(method, ispos):
    ref_L = surface.load_surf_data(
        '../atlases/fsaverage.L.Glasser.32k_fs_LR.label.gii'
    ).astype(np.float32)
    atlas_mask_L = ref_L > 0

    ref_R = surface.load_surf_data(
        '../atlases/fsaverage.R.Glasser.32k_fs_LR.label.gii'
    ).astype(np.float32)
    atlas_mask_R = ref_R > 0

    neigh_L = surface.load_surf_data(
        '../atlases/fsaverage.L.Glasser_search_light_r20mm.32k_fs_LR.func.gii'
    ).astype(bool)
    neigh_L = neigh_L[atlas_mask_L,:]

    neigh_R = surface.load_surf_data(
        '../atlases/fsaverage.R.Glasser_search_light_r20mm.32k_fs_LR.func.gii'
    ).astype(bool)
    neigh_R = neigh_R[atlas_mask_R,:]

    print(neigh_L.shape, neigh_R.shape)

    sub_lists = [
        ('../sub_lists/sub_list_adni_nc_diag2_drt.txt', 'NC'),
        ('../sub_lists/sub_list_adni_mci_diag_drt.txt', 'MCI'),
        ('../sub_lists/sub_list_adni_ad_drt.txt', 'AD')
    ]

    tmp = np.load('adni/parc_loading_tval_{}.{}.npy'.format(method, '_'.join([kk[1] for kk in sub_lists])), allow_pickle=True).item()

    if ispos:
        pv_ind = (np.array(tmp['pv']) < 0.05) * (np.array(tmp['tv']) > 0)
    else:
        pv_ind = (np.array(tmp['pv']) < 0.05) * (np.array(tmp['tv']) < 0)
    print("pv_ind: ", pv_ind.shape)

    # topo = np.load('adni/parc_loading_{}.{}.npy'.format(
    topo = np.load('adni/parc_norm_loading_{}.{}.npy'.format(
        method, '_'.join([kk[1] for kk in sub_lists])
    ))
    topo = topo[:, pv_ind].sum(1) # Subj
    topo -= topo.min()

    topo_all = []
    group_name = []
    cnt = 0
    for i, (sub_list_name, brev) in enumerate(sub_lists):
        sub_list = np.loadtxt(sub_list_name, dtype=str)
        group_name.extend([brev] * sub_list.size)
        topo_all.append(topo[cnt:cnt+sub_list.size])
        cnt += sub_list.size

    from scipy import stats
    _, p1 = stats.ttest_ind(topo_all[0], topo_all[1])
    _, p2 = stats.ttest_ind(topo_all[1], topo_all[2])
    _, p3 = stats.ttest_ind(topo_all[0], topo_all[2])
    print('func.', p1 * 3, p2 * 3, p3 * 3)

    sns.set_theme(style="ticks", palette='pastel', font_scale=1.3)
    plt.rcParams['figure.figsize'] = (4, 5)

    df = pd.DataFrame({
        'loading': topo,
        'group': group_name,
    })
    print(topo.mean(), topo.std(), topo.min(), topo.max())

    ax = sns.barplot(data=df, x='group', y='loading',
                     palette="Set2")
    ax.set(ylim=(topo.mean() - topo.std(), topo.mean() + topo.std()), ylabel=None, yticklabels=[])  # remove the axis label
    ax.set(xlabel=None)  # remove the axis label

    sns.despine(left=False, right=True, bottom=False, top=True)
    plt.savefig('adni/barplot_{}_{}_{}_scores.tif'.format(method, 'pos' if ispos else 'neg', '_'.join([kk[1] for kk in sub_lists])))
    plt.close()


def regress_plot_multi_cov(method, ispos, group=['NC','MCI','AD']):

    sub_lists = []
    if 'NC' in group:
        sub_lists.append(('../sub_lists/sub_list_adni_nc_diag2_drt.txt', 'NC'))
    if 'MCI' in group:
        sub_lists.append(('../sub_lists/sub_list_adni_mci_diag_drt.txt', 'MCI'))
    if 'AD' in group:
        sub_lists.append(('../sub_lists/sub_list_adni_ad_drt.txt', 'AD'))

    # score_names = [
    #     'ADA13', 'CDRSB', 'FAQ', 'MMSE', 'MOCA'
    # ]
    score_names = [
        'ADA13', 'FAQ', 'MOCA'
    ]

    sns.set_theme(style="ticks", palette='pastel', font_scale=1.3)

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))

    for i in range(nrows * ncols):
        if i < len(score_names):
            score_name = score_names[i]

            tmp = np.load(
                'adni/regress_{}_{}_{}_{}.npy'.format('pos' if ispos else 'neg', method, '_'.join([kk[1] for kk in sub_lists]), score_name),
                allow_pickle=True).item()
            type = np.asarray(tmp['type'])

            tmp['topo_vali'] = np.asarray(tmp['topo_vali'])
            tmp['sex'] = np.asarray(tmp['sex'])
            tmp['age'] = np.asarray(tmp['age'])
            tmp['rms'] = np.asarray(tmp['rms'])
            tmp['etiv'] = np.asarray(tmp['etiv'])
            tmp['y'] = np.asarray(tmp['y'])

            smsv = (tmp['topo_vali'] - np.mean(tmp['topo_vali'])) / np.std(tmp['topo_vali'])
            valiind = (smsv > -3) & (smsv < 3)
            # print(valiind)
            tmp['topo_vali'] = tmp['topo_vali'][valiind]
            tmp['sex'] = tmp['sex'][valiind]
            tmp['age'] = tmp['age'][valiind]
            tmp['rms'] = tmp['rms'][valiind]
            tmp['etiv'] = tmp['etiv'][valiind]
            tmp['y'] = tmp['y'][valiind]
            type = type[valiind]

            X = pd.DataFrame({
                'x': tmp['topo_vali'],
                'sex': tmp['sex'],
                'age': tmp['age'],
                'rms': tmp['rms'],
                'etiv': tmp['etiv'],
            })
            # print(df1)
            X = pd.get_dummies(X, columns=['sex'], drop_first=True)
            X = sm.add_constant(X)
            X = X.astype(float)
            # print(X)

            y = tmp['y']
            model = sm.OLS(
                y, X,
                # family=sm.families.Gaussian()
                )
            result = model.fit()

            y_adj = y - result.params[0] \
                      - result.params[2] * X['age'] \
                      - result.params[3] * X['rms'] \
                      - result.params[4] * X['etiv'] \
                      - result.params[5] * X['sex_M']
            y_adj = (y_adj - y_adj.mean()) / y_adj.std()

            # r_value = result.rsquared ** 0.5
            # pv = result.pvalues[1]
            # r_value, pv = pearsonr(X['x'], y)
            # print('r = {:.2f}, p = {:.1e}'.format(r_value, pv))

            # y_adj = y

            r_value, pv = pearsonr(X['x'], y_adj)
            print('reg. r = {:.2f}, p = {:.1e}'.format(r_value, pv))

            df = pd.DataFrame({
                'x': np.array(X['x']),
                'y': np.array(y_adj),
                'type': type
            })
            sns.regplot(
                data=df,
                x='x', y='y',
                scatter=False, line_kws=dict(color="k"), ax=axes[i // ncols, i % ncols]
            )

            sns.scatterplot(
                x='x', y='y', hue='type', data=df,
                legend=False, palette="Set2", ax=axes[i // ncols, i % ncols]
            )

            if pv >= 0.05:
                psd = 'p > 0.05'
            elif pv >= 0.01:
                psd = 'p < 0.05'
            elif pv >= 0.001:
                psd = 'p < 0.01'
            else:
                psd = 'p < 0.001'

            axes[i // ncols, i % ncols].set(title='{}\nr = {:.2f}, {}'.format(score_name, r_value, psd))
            axes[i // ncols, i % ncols].set(xlabel="", xticklabels=[], ylabel="")

        else:
            axes[i // ncols, i % ncols].axis('off')

    sns.despine(left=False, right=True, bottom=False, top=True)
    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    plt.savefig('adni/lmplot_{}_{}_{}_scores.tif'.format(method, 'pos' if ispos else 'neg', '_'.join([kk[1] for kk in sub_lists])))
    plt.close()


def print_tv_pv(method, ispos=True):
    tmp = np.load('adni/parc_loading_tval_{}.{}.npy'.format(method, 'NC_MCI_AD'),
                  allow_pickle=True).item()
    if ispos:
        pv_ind = (np.array(tmp['pv']) < 0.05) * (np.array(tmp['tv']) > 0)
    else:
        pv_ind = (np.array(tmp['pv']) < 0.05) * (np.array(tmp['tv']) < 0)

    parcel_name = np.loadtxt('Glasser360_list.txt', dtype=str)

    pv_indss = np.where(pv_ind)[0]
    print(pv_indss)
    print(np.array(tmp['tv'])[pv_indss])

    ordp = np.argsort(np.array(tmp['tv'])[pv_indss])
    for pp in pv_indss[ordp]:
        print("{}\t{}\t{:.2f}\t{:.1e}".format(pp+1, parcel_name[pp], tmp['tv'][pp], tmp['pv'][pp]))


def print_sex_group_mmse_cdr(method):
    from scipy.stats import f_oneway
    sex_list = np.loadtxt('adni/NC_MCI_AD_sex.txt', dtype=str)
    age_list = np.loadtxt('adni/NC_MCI_AD_age.txt', dtype=str)
    mmse_list = np.loadtxt('adni/ADNI_NC_MCI_AD_MMSE.txt', dtype=str)
    cdr_list = np.loadtxt('adni/ADNI_NC_MCI_AD_CDRSB.txt', dtype=str)
    ada_list = np.loadtxt('adni/ADNI_NC_MCI_AD_ADA13.txt', dtype=str)
    faq_list = np.loadtxt('adni/ADNI_NC_MCI_AD_FAQ.txt', dtype=str)
    moca_list = np.loadtxt('adni/ADNI_NC_MCI_AD_MOCA.txt', dtype=str)


    dicsex, dicage, dicmmse, diccdr, dicada, dicfaq, dicmoca  = {}, {}, {}, {}, {}, {}, {}
    for ss in sex_list:
        dicsex[ss[0]] = ss[1]
    for ss in age_list:
        dicage[ss[0]] = int(ss[1])
    for ss in mmse_list:
        if ss[1] != 'nan':
            dicmmse[ss[0]] = float(ss[1])
    for ss in cdr_list:
        if ss[1] != 'nan':
            diccdr[ss[0]] = float(ss[1])
    for ss in ada_list:
        if ss[1] != 'nan':
            dicada[ss[0]] = float(ss[1])
    for ss in faq_list:
        if ss[1] != 'nan':
            dicfaq[ss[0]] = float(ss[1])
    for ss in moca_list:
        if ss[1] != 'nan':
            dicmoca[ss[0]] = float(ss[1])

    sub_lists = [
        ('../sub_lists/sub_list_adni_nc_diag2_drt.txt', 'NC'),
        ('../sub_lists/sub_list_adni_mci_diag_drt.txt', 'MCI'),
        ('../sub_lists/sub_list_adni_ad_drt.txt', 'AD')
    ]

    group_sex, group_age, group_mmse, group_cdr, group_ada, group_faq, group_moca = [], [], [], [], [], [], []
    for j, (sub_list_name, brev) in enumerate(sub_lists):
        sex, age, mmse, cdr, ada, faq, moca = [], [], [], [], [], [], []

        sub_list = np.loadtxt(sub_list_name, dtype=str)
        for sub in sub_list:
            if sub in dicsex.keys():
                sex.append(dicsex[sub] == 'F')

            if sub in dicage.keys():
                age.append(dicage[sub])

            if sub in dicmmse.keys():
                mmse.append(dicmmse[sub])

            if sub in diccdr.keys():
                cdr.append(diccdr[sub])

            if sub in dicada.keys():
                ada.append(dicada[sub])
            if sub in dicfaq.keys():
                faq.append(dicfaq[sub])
            if sub in dicmoca.keys():
                moca.append(dicmoca[sub])

        sex = np.array(sex)
        age = np.array(age)
        mmse = np.array(mmse)
        cdr = np.array(cdr)
        ada = np.array(ada)
        faq = np.array(faq)
        moca = np.array(moca)

        print('n = {};M = {}, F = {};{:.1f} ± {:.2f};{}-{};{:.1f} ± {:.2f};{:.1f} ± {:.2f};{:.1f} ± {:.2f}'.format(
            sex.size,
            sex.size - sex.sum(), sex.sum(),
            age.mean(), age.std(),
            age.min(), age.max(),
            # mmse.mean(), mmse.std(),
            # cdr.mean(), cdr.std()
            ada.mean(), ada.std(),
            faq.mean(), faq.std(),
            moca.mean(), moca.std()
        ))

        group_sex.append(sex)
        group_age.append(age)
        # group_mmse.append(mmse)
        # group_cdr.append(cdr)
        group_ada.append(ada)
        group_faq.append(faq)
        group_moca.append(moca)

    fst, pv = f_oneway(*group_sex)
    print('{:.2f};{:.1e}'.format(fst, pv))
    fst, pv = f_oneway(*group_age)
    print('{:.2f};{:.1e}'.format(fst, pv))
    # fst, pv = f_oneway(*group_mmse)
    # print('{:.2f};{:.1e}'.format(fst, pv))
    # fst, pv = f_oneway(*group_cdr)
    # print('{:.2f};{:.1e}'.format(fst, pv))
    fst, pv = f_oneway(*group_ada)
    print('{:.2f};{:.1e}'.format(fst, pv))
    fst, pv = f_oneway(*group_faq)
    print('{:.2f};{:.1e}'.format(fst, pv))
    fst, pv = f_oneway(*group_moca)
    print('{:.2f};{:.1e}'.format(fst, pv))


def indiv_main():
    method = 'Unet_indiv_msm_dr_lam10_Glasser_drt_fold1'
    # method = 'Unet_indiv_msm_dr_lam10_Glasser_drt_fold2'

    # glm_size_neigh(method)
    # corr_loading_parc_size(method)

    # bar_plot_3groups(method, ispos=False)
    # bar_plot_3groups(method, ispos=True)

    # corr_size_score(method, ispos=False, group=['NC','MCI','AD'])
    # corr_size_score(method, ispos=True, group=['NC','MCI','AD'])
    #
    # regress_plot_multi_cov(method, ispos=False, group=['NC','MCI','AD'])
    # regress_plot_multi_cov(method, ispos=True, group=['NC','MCI','AD'])

    # corr_size_score(method, ispos=False, group=['MCI','AD'])
    # corr_size_score(method, ispos=True, group=['MCI','AD'])
    #
    # regress_plot_multi_cov(method, ispos=False, group=['MCI','AD'])
    # regress_plot_multi_cov(method, ispos=True, group=['MCI','AD'])

    print('NC')
    corr_size_score(method, ispos=False, group=['NC'])
    corr_size_score(method, ispos=True, group=['NC'])

    regress_plot_multi_cov(method, ispos=False, group=['NC'])
    regress_plot_multi_cov(method, ispos=True, group=['NC'])

    print('MCI')
    corr_size_score(method, ispos=False, group=['MCI'])
    corr_size_score(method, ispos=True, group=['MCI'])

    regress_plot_multi_cov(method, ispos=False, group=['MCI'])
    regress_plot_multi_cov(method, ispos=True, group=['MCI'])

    # print_tv_pv(method, ispos=False)
    # print_tv_pv(method, ispos=True)


def group_main():
    method = 'group_Glasser'

    glm_group_size(method)
    corr_loading_parc_size(method)

    bar_plot_3groups(method, ispos=False)
    bar_plot_3groups(method, ispos=True)

    # corr_size_score(method, ispos=False, group=['NC','MCI','AD'])
    # corr_size_score(method, ispos=True, group=['NC','MCI','AD'])
    #
    # regress_plot_multi_cov(method, ispos=False, group=['NC','MCI','AD'])
    # regress_plot_multi_cov(method, ispos=True, group=['NC','MCI','AD'])

    corr_size_score(method, ispos=False, group=['MCI','AD'])
    corr_size_score(method, ispos=True, group=['MCI','AD'])

    regress_plot_multi_cov(method, ispos=False, group=['MCI','AD'])
    regress_plot_multi_cov(method, ispos=True, group=['MCI','AD'])

    # print_tv_pv(method, ispos=False)
    # print_tv_pv(method, ispos=True)


if __name__ == '__main__':
    method = 'Unet_indiv_msm_dr_lam10_Glasser_drt_fold1'

    # corr_loading_parc_size(method)
    indiv_main()
    # group_main()
