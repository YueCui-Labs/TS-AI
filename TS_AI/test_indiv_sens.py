#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:46:50 2019

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com
"""

import torch
import torch.nn.functional as F

import argparse
import torchvision
import numpy as np

import os, subprocess

from tqdm import tqdm

import nibabel as nib
from sphericalunet.utils.vtk import read_vtk
from nilearn import surface
import scipy
from model import Unet_indiv
from config import parse_args
from dataset import unet_indiv_dataset
from utils import AverageMeter

from sys import stdout

import scipy.io as scio


def save_indiv(template, prediction, group_label_32k, prename):
    indiv_label_40k = '{}.40k.label.gii'.format(prename)
    indiv_label_32k = '{}.32k_fs_LR.label.gii'.format(prename)

    template.remove_gifti_data_array(0)
    template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.array(prediction).astype(np.int32)))
    nib.loadsave.save(template, indiv_label_40k)

    p = subprocess.Popen('wb_command -label-resample {} {} {} BARYCENTRIC {}\n'.format(
        indiv_label_40k, sphere_40k, sphere_32k, indiv_label_32k),
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return_code = p.wait()
    if return_code != 0:
        print('wb_command -label-resample {} {} {} BARYCENTRIC {}\n'.format(
            indiv_label_40k, sphere_40k, sphere_32k, indiv_label_32k)
        )
        print('wb_command -label-resample failed')
        exit(1)

    p = subprocess.Popen("wb_command -label-mask {} {} {}".format(
        indiv_label_32k, group_label_32k, indiv_label_32k),
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return_code = p.wait()
    if return_code != 0:
        print('wb_command -label-mask failed')

        exit(1)

    p = subprocess.Popen('rm -rf {}'.format(indiv_label_40k),
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return_code = p.wait()
    if return_code != 0:
        print('rm -rf failed')
        exit(1)


def save_recon(template, prediction, group_label_32k, prename):
    pred_40k = '{}.40k.func.gii'.format(prename)
    pred_32k = '{}.32k_fs_LR.func.gii'.format(prename)
    template.remove_gifti_data_array(0)
    template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.array(prediction).astype(np.float32)))
    nib.loadsave.save(template, pred_40k)

    p = subprocess.Popen('wb_command -metric-resample {} {} {} BARYCENTRIC {}\n'.format(
        pred_40k, sphere_40k, sphere_32k, pred_32k),
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return_code = p.wait()
    if return_code != 0:
        print('wb_command -metric-resample {} {} {} BARYCENTRIC {}\n'.format(
            pred_40k, sphere_40k, sphere_32k, pred_32k))
        print('metric-resample failed!')
        exit(1)

    p = subprocess.Popen("wb_command -metric-mask {} {} {}".format(
        pred_32k, group_label_32k, pred_32k),
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return_code = p.wait()
    if return_code != 0:
        print('metric-mask failed!')
        exit(1)

    p = subprocess.Popen('rm -rf {}'.format(pred_40k),
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return_code = p.wait()
    if return_code != 0:
        print('rm failed!')
        exit(1)


def save_label_met(template, prediction, group_label_32k, prename):
    pred_40k = '{}.40k.func.gii'.format(prename)
    pred_32k = '{}.32k_fs_LR.func.gii'.format(prename)
    template.remove_gifti_data_array(0)
    template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.array(prediction).astype(np.float32)))
    nib.loadsave.save(template, pred_40k)

    p = subprocess.Popen('wb_command -metric-resample {} {} {} BARYCENTRIC {} -largest\n'.format(
        pred_40k, sphere_40k, sphere_32k, pred_32k),
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return_code = p.wait()
    if return_code != 0:
        print('wb_command -metric-resample {} {} {} BARYCENTRIC {} -largest\n'.format(
            pred_40k, sphere_40k, sphere_32k, pred_32k))
        print('metric-resample failed!')
        exit(1)

    p = subprocess.Popen("wb_command -metric-mask {} {} {}".format(
        pred_32k, group_label_32k, pred_32k),
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return_code = p.wait()
    if return_code != 0:
        print('metric-mask failed!')
        exit(1)

    p = subprocess.Popen('rm -rf {}'.format(pred_40k),
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return_code = p.wait()
    if return_code != 0:
        print('rm failed!')
        exit(1)


def evaluate(logits, labels, recon, feats):
    """
    :param logits: (B,K,V)
    :param labels: (B,V)
    :return:  ov (1),
              nparc (1),
              pearsonr (
    """
    ov, nparc = 0, 0
    B = logits.shape[0]

    for logit, label in zip(logits, labels):  # (K,V) (V)
        indiv_label = logit[:, label > 0].argmax(axis=0) + 1  # (Vk)
        group_label = label[label > 0]  # (Vk)
        ov += torch.sum(indiv_label == group_label).float() / group_label.numel()
        nparc += indiv_label.unique().numel()
    ov /= B
    nparc /= B

    pearsonr = []
    for i, (rec, fea) in enumerate(zip(recon, feats)):
        pearsonr.append(corrcoef(rec.float().view(-1, logits.shape[2]), fea.float().view(-1, logits.shape[2])))

    return (ov.item(), nparc, *pearsonr)


def corrcoef(tensor1, tensor2):
    """
    :param tensor1: (J,V)
    :param tensor2: (J,V)
    :return:
    """
    vx = tensor1 - tensor1.mean(1, keepdim=True)  # (J,V)
    vy = tensor2 - tensor2.mean(1, keepdim=True)  # (J,V)
    # print(vx.isnan().sum(), vy.isnan().sum())
    cost = (vx * vy).sum(1) / ((vx ** 2).sum(1).sqrt() * (vy ** 2).sum(1).sqrt() + 1e-4)  # (J)
    # print(cost.isnan().sum())
    return cost.mean().item()


def soft_dice_parc(soft_label_0, soft_label_turb, label_num):
    """

    :param soft_label_0: (V,K)
    :param soft_label_turb:
    :param label_num:
    :return:
    """
    # sil = []
    # print(soft_label_0.shape, soft_label_turb.shape)
    # for ll in range(label_num):
    #     dice = 2 * (soft_label_0[:,ll] * soft_label_turb[:,ll]).sum() / ( soft_label_0[:,ll].sum() + soft_label_turb[:,ll].sum() + 1e-4)
    #     sil.append(dice)
    sil = (2 * (soft_label_0 * soft_label_turb).sum(0) + 1e-5) / (soft_label_0.sum(0) + soft_label_turb.sum(0) + 1e-5)
    return sil


args = parse_args()

reference_atlas = args.reference_atlas
hemi = args.hemi
feat_dir = args.feat_dir

modals = args.modals
in_chs = {}
if 'r' in args.modals:
    in_chs['r'] = 47
if 't' in args.modals:
    in_chs['t'] = 24
    # in_chs['t'] = 47
if 'd' in args.modals:
    in_chs['d'] = 41
if 'g' in args.modals:
    in_chs['g'] = 50

label_num = args.out_channels

save_dir = args.save_dir

test_sub_list_file = args.test_sub_list_file
indiv_label_dir = args.indiv_label_dir
# pred_dir = args.feat_dir


sphere_32k = args.sphere_32k
sphere_40k = args.sphere_40k
sphere_10k = "./neigh_indices/Sphere.10242.surf.gii"
sphere_3k = "./neigh_indices/Sphere.2562.surf.gii"
sphere_642 = "./neigh_indices/Sphere.642.surf.gii"

r_max = 32
n_warp = 100

lam_recon = int(args.lamb)

syn_task = args.syn_task_name
prename = args.indiv_prename
print(prename)

model_dir = '{}_fold{}'.format(
    prename, args.fold
)

print(model_dir)

batch_size = 1

if __name__ == "__main__":
    label_32k_file = 'atlases/fsaverage.{}.{}.32k_fs_LR.label.gii'.format(hemi, reference_atlas)
    label_40k_file = 'atlases/fsaverage.{}.{}.40k.label.gii'.format(hemi, reference_atlas)
    label_10k_file = 'atlases/fsaverage.{}.{}.10k.label.gii'.format(hemi, reference_atlas)
    label_3k_file = 'atlases/fsaverage.{}.{}.3k.label.gii'.format(hemi, reference_atlas)
    label_642_file = 'atlases/fsaverage.{}.{}.642.label.gii'.format(hemi, reference_atlas)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)

    test_dataset = unet_indiv_dataset(
        test_sub_list_file,
        modals,
        feat_dir,
        label_40k_file,
        hemi,
        syn_task,
        transform=False,
        istrain=False,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers
    )

    label2feat = []
    label2mod = []
    vsal = 0.0
    cnt = 0

    model = Unet_indiv(modals, in_chs, label_num, istrain=False)
    model.to(device)
    print("model have been sent to device")
    stdout.flush()

    # load model parameter
    save_point_file = os.path.join(save_dir, model_dir, 'model_best.pth.tar')

    if os.path.exists(save_point_file):
        checkpoint_params = torch.load(save_point_file)

        start_epoch = checkpoint_params['epoch']
        model.load_state_dict(checkpoint_params['model'])

        print('best model loaded, epoch: %d' % start_epoch)
        stdout.flush()
    else:
        raise NotImplementedError('model parameter not found!')

    # template label 40k file
    label_40k = nib.load(label_40k_file)

    label_out = surface.load_surf_data(label_40k_file)  # (#vert,)
    label_out[label_out < 0] = 0  # there is area = -1
    label_index_lookup = np.unique(label_out[label_out > 0])

    label_out_cuda = torch.Tensor(label_out).to(device)

    val_metrics = AverageMeter()  # ov, nparc, rv, mae

    for feats, label, sub in tqdm(test_dataloader):

        # print(sub[0])
        # if cnt > 2:
        #     break
        # else:
        #     cnt+=1
        cnt+=1

        model.eval()
        B = label.shape[0]

        feats = [f.to(device) for f in feats]
        label = label.to(device)

        idv, _, _, _ = model(feats)

        idv = idv.squeeze(dim=0).detach().T

        idv_pos = idv[label_out_cuda > 0, :]  # (V,K)
        idv_soft = F.softmax(idv_pos, dim=1)  # (V,K)

        indiv_label = idv.argmax(axis=1)  # (V)

        ### label2feat
        dice_to_feat = []
        dice_self_idv = soft_dice_parc(idv_soft, idv_soft, label_num)

        for mi in range(len(feats)):
            for fj in range(feats[mi].shape[1]):
                feats_trub = [f.clone() for f in feats]
                feats_trub[mi][:,fj,:] = 0
                # feats_trub[mi][:,fj,:] = avg_feat[mi][None, fj, :]
                idv_turb, _, _, _ = model(feats_trub)

                idv_turb = idv_turb.squeeze(dim=0).detach().T

                idv_turb_pos = idv_turb[label_out_cuda > 0, :]
                idv_turb_pos_soft = F.softmax(idv_turb_pos, dim=1)  # (V,K)
                dice_to_feat.append(soft_dice_parc(idv_soft, idv_turb_pos_soft, label_num))

                # idv_turb_label = idv_turb.argmax(axis=1)  # (1,V)
                # dice_to_feat.append(dice_parc(indiv_label, idv_turb_label, label_num))

        dice_to_feat = torch.stack(dice_to_feat, dim=1)  # (K, M)
        dice_to_feat = (dice_to_feat / dice_self_idv[:, None]).cpu().numpy()  # (K, M)
        label2feat.append(dice_to_feat)

        ### label2mod
        dice_to_mod = []
        for mi in range(len(feats)):
            feats_trub = [f.clone() for f in feats]
            feats_trub[mi][:, :, :] = 0
            # feats_trub[mi][:,:,:] = avg_feat[mi][None, :, :]
            idv_turb, _, _, _ = model(feats_trub)

            idv_turb = idv_turb.squeeze(dim=0).detach().T

            idv_turb_pos = idv_turb[label_out_cuda > 0, :]
            idv_turb_pos_soft = F.softmax(idv_turb_pos, dim=1)  # (V,K)
            dice_to_mod.append(soft_dice_parc(idv_soft, idv_turb_pos_soft, label_num))

        dice_to_mod = torch.stack(dice_to_mod, dim=1)  # (K, M)
        dice_to_mod = (dice_to_mod / dice_self_idv[:, None]).cpu().numpy()  # (K, M)
        label2mod.append(dice_to_mod)

        # np.save(
        #     os.path.join(indiv_label_dir, sub[0], prename + '_sens.npy'),
        #     {'dice_to_feat':dice_to_feat, 'dice_to_mod':dice_to_mod}
        # )

    # ### label2feat
    # label2feat = np.stack(label2feat, axis=0)
    # np.save('{}/{}/label2feat_soft_avg_{}.npy'.format(save_dir, model_dir, hemi), label2feat)
    #
    # label2feat = label2feat.mean(0)
    # idv_sali = np.zeros((label_out.shape[0], label2feat.shape[1]))
    #
    # for i, ll in enumerate(label_index_lookup):
    #     idv_sali[label_out == ll, :] = label2feat[i, :]
    # print(idv_sali.shape)
    #
    # save_label_met(label_40k, idv_sali, label_32k_file,
    #            prename=os.path.join(save_dir, model_dir, 'avg_label2feat_soft_avg_{}'.format(hemi))
    #            )
    #
    label2mod = np.stack(label2mod, axis=0)
    np.save('{}/{}/label2mod_soft_avg_{}.npy'.format(save_dir, model_dir, hemi), label2mod)

    ### label2mod
    label2mod = label2mod.mean(0)
    idv_sali = np.zeros((label_out.shape[0], label2mod.shape[1]))

    for i, ll in enumerate(label_index_lookup):
        idv_sali[label_out == ll, :] = label2mod[i, :]
    print(idv_sali.shape)

    save_label_met(label_40k, idv_sali, label_32k_file,
               prename=os.path.join(save_dir, model_dir, 'avg_label2mod_soft_avg_{}'.format(hemi))
               )
