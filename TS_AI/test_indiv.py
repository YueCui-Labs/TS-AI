#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:46:50 2019

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com
"""

import torch
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


def dice_parc(label_0, label_turb, label_num):
    # label_index_lookup = np.unique(label_out[label_out > 0])
    sil = []
    # print(2 * (label_0 == label_turb).sum() / (label_0.size + label_turb.size))
    for ll in range(label_num):
        dice = 2 * ((label_0 == ll) & (label_turb == ll)).sum() / ( (label_0 == ll).sum() + (label_turb == ll).sum() + 10e-4)
        # print(((label_0 == ll) & (label_turb == ll)).sum(), (label_0 == ll).sum() + (label_turb == ll).sum(), dice)
        sil.append(dice)
    return np.array(sil)


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

    val_metrics = AverageMeter()  # ov, nparc, rv, mae

    salient = []
    vsal = 0.0
    cnt = 0
    for feats, label, sub in tqdm(test_dataloader):
        if not os.path.exists('{}.32k_fs_LR.label.gii'.format(os.path.join(indiv_label_dir, sub[0], prename))):

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

            idv, recon, _, _ = model(feats)

            idv = idv.squeeze(dim=0).detach().cpu().T
            # if not os.path.exists(os.path.join(indiv_label_dir, sub[0])):
            #     os.mkdir(os.path.join(indiv_label_dir, sub[0]))
            # if not os.path.exists(os.path.join(indiv_label_dir, sub[0], prename + '_prob.32k_fs_LR.func.gii')):
            #     save_recon(label_40k, idv, label_32k_file,
            #                prename=os.path.join(indiv_label_dir, sub[0], prename + '_prob')
            #                )
            # idv_pos = idv[label_out > 0, :]
            # idv_size = idv_pos.sum(0)  # (K)
            # idv_soft_size = scipy.special.softmax(idv_pos, axis=1).sum(0)  # (K)
            # np.save(os.path.join(indiv_label_dir, sub[0], prename + '_size.npy'),
            #         {"idv_size": idv_size, "idv_soft_size": idv_soft_size})

            # ------------------------------
            indiv_label = idv.argmax(axis=1)  # (1,V)
            indiv_label = indiv_label.detach().cpu()
            label = label.cpu()

            # ------------------------------
            label_uni = np.zeros(label.shape)

            for ch in range(label_num):
                label_uni[(label > 0) & (indiv_label == ch)] = label_index_lookup[ch]
            label_uni = label_uni.reshape(-1)

            os.makedirs(os.path.join(indiv_label_dir, sub[0]), exist_ok=True)
            save_indiv(label_40k, label_uni, label_32k_file,
                       prename=os.path.join(indiv_label_dir, sub[0], prename)
                       )
