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

from model import Unet_syn
from config import parse_args
from dataset import unet_syn_dataset
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

    # p = subprocess.Popen('rm -rf {}'.format(pred_40k),
    #                      shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # return_code = p.wait()
    # if return_code != 0:
    #     print('rm failed!')
    #     exit(1)


def evaluate(labels, recon, feats):
    """
    :param recon, feats: (B,K,V)
    :param labels: (B,V)
    """

    pearsonr = corrcoef(recon.float().permute(1,0,2)[:,labels>0],
                        feats.float().permute(1,0,2)[:,labels>0])

    return pearsonr


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


args = parse_args()

reference_atlas = args.reference_atlas
hemi = args.hemi
feat_dir = args.feat_dir

modals = args.modals
out_modal = args.out_modal

in_chs = {'r': 47, 't': 24, 'd': 41, 'g': 50}
# in_chs = {'r': 47, 't': 47, 'd': 41, 'g': 50}

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

syn_task = '{}.{}'.format(args.syn_task_name, hemi)
syn_prename = args.syn_prename

model_dir = '{}_fold{}'.format(syn_prename, args.fold)
print(model_dir)

batch_size = 1

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)

    label_32k_file = 'atlases/fsaverage.{}.{}.32k_fs_LR.label.gii'.format(hemi, reference_atlas)
    label_40k_file = 'atlases/fsaverage.{}.{}.40k.label.gii'.format(hemi, reference_atlas)

    test_dataset = unet_syn_dataset(
        test_sub_list_file,
        modals,
        out_modal,
        feat_dir,
        label_40k_file,
        hemi,
        transform=False,
        istrain=False
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers
    )

    model = Unet_syn(modals, out_modal, in_chs)

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

    recons = 0
    # recon = surface.load_surf_data(os.path.join(save_dir, 'avg_recon24.'+hemi+'.40k.func.gii'))  # (#vert,)
    # recon = torch.Tensor(recon[None,:,:]).to(device)
    # print(recon.shape)

    cnt = 0

    for feats, label, sub in tqdm(test_dataloader):
    # for feats, out_feat, label, sub in tqdm(test_dataloader):
        if not os.path.exists('{}.32k_fs_LR.func.gii'.format(os.path.join(feat_dir, sub[0], syn_task))):
            model.eval()
            # B = label.shape[0]
            # recons += feats[2][0,...].numpy()
            # if cnt>=2:
            #     break
            # else:
            #     cnt+=1

            # recons += out_feat[0]

            feats = [f.to(device) for f in feats]
            # out_feat = out_feat.to(device)
            label = label.to(device)

            recon = model(feats)
            # pr = evaluate(label, recon, out_feat)
            # val_metrics.update(pr, n=1)


            save_recon(label_40k, recon[0,...].T.detach().cpu().numpy(), label_32k_file,
                       prename=os.path.join(feat_dir, sub[0], syn_task)
                       )

    print('<Validation>  PearsonR: {pearsonr:.4f}'.format(
        pearsonr=val_metrics.avg
    ))

    # recons /= len(test_dataloader)
    #
    # save_recon(label_40k, recons.T, label_32k_file,
    #            prename=os.path.join(save_dir, 'avg_recon47.'+hemi)
    #            )
