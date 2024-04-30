# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torchvision
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
import shutil
from sys import stdout

from model import Unet_syn
from config import parse_args
from dataset import unet_syn_dataset
from loss import syn_loss
from utils import AverageMeter

from einops import rearrange

# from tensorboardX import SummaryWriter
# writer = SummaryWriter('log/a')

# Half precison
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True


def train_one_step(feats, out_feat):
    """
    :param feats: [(B, C, V)]i=1:M
    :return:
    """
    model.train()
    feats = [f.to(device) for f in feats]
    out_feat = out_feat.to(device)

    # print([f.shape for f in feats], masks.shape)
    # with autocast():
    recon = model(feats)
    loss_tot = criterion(recon, out_feat)

    scaler.scale(loss_tot).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    metrics = evaluate(recon, out_feat)
    return metrics, loss_tot.item()


def evaluate(recon, feats):
    """
    :param label: (B,V)
    :param recon, feats: (B,Ci,V)
    :return:  ov (1),
              nparc (1),
    """

    return corrcoef(recon.float().reshape(-1, recon.shape[2]), feats.float().reshape(-1, feats.shape[2]))


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


def val_during_training(dataloader):
    model.eval()
    val_losses = AverageMeter()  # total = idv, recon, mc, cc
    val_metrics = AverageMeter()  # ov, nparc, rv, mae

    for feats, out_feat, label, _ in dataloader:
        # data: ((B,C_i, V))i=1:M
        # labels: (B, V)
        feats = [f.to(device) for f in feats]
        out_feat = out_feat.to(device)

        with torch.no_grad():
            recon = model(feats)
            loss_tot = criterion(recon, out_feat)
            metrics = evaluate(recon, out_feat)

        val_losses.update(loss_tot.item(), n=B)
        val_metrics.update(metrics, n=B)

    return val_losses.avg, val_metrics.avg


def plotCurve(x_vals, y_vals, name,
              x2_vals=None, y2_vals=None,
              x_label='epoch', y_label='loss', legend=None,
              figsize=(3.5, 2.5)):
    # set figsize
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')

    if legend:
        plt.legend(legend)

    plt.savefig(name)
    plt.close()


args = parse_args()

reference_atlas = args.reference_atlas
hemi = args.hemi
feat_dir = args.feat_dir

modals = args.modals
out_modal = args.out_modal

in_chs = {'r': 47, 't': 24, 'd': 41, 'g': 50}

label_num = args.out_channels

save_dir = args.save_dir

train_sub_list_file = args.train_sub_list_file
vali_sub_list_file = args.vali_sub_list_file
batch_size = args.batch_size

learning_rate = args.learning_rate * args.batch_size
epoch_num = args.epoch_num
save_epoch_step = epoch_num - 1

r_max = 32
n_warp = 100

base_dir = os.path.join(
    save_dir,
    '{}_fold{}'.format(args.syn_prename, args.fold)
)

def report_gpu():
    # print(torch.cuda.list_gpu_processes())
    # gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # label_32k_file = 'atlases/fsaverage.{}.{}.32k_fs_LR.label.gii'.format(hemi, reference_atlas)
    label_40k_file = 'atlases/fsaverage.{}.{}.40k.label.gii'.format(hemi, reference_atlas)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)
    stdout.flush()

    train_dataset = unet_syn_dataset(
        train_sub_list_file,
        args.modals,
        out_modal,
        feat_dir,
        label_40k_file,
        hemi,
        transform=False,
        r_max=r_max,
        n_warp=n_warp,
    )
    val_dataset = unet_syn_dataset(
        vali_sub_list_file,
        args.modals,
        out_modal,
        feat_dir,
        label_40k_file,
        hemi,
        transform=False,
        r_max=r_max,
        n_warp=n_warp
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers
    )

    model = Unet_syn(modals, out_modal, in_chs)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    model.to(device)
    print("model have been sent to device")

    criterion = syn_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=1, verbose=True,
    #                                                        threshold=0.0001, threshold_mode='rel', min_lr=0.000001)

    train_ov_list, vali_ov_list = [], []
    train_nparc_list, vali_nparc_list = [], []

    start_epoch, best_vali_metric = 0, 0

    # loading checkpoints from end to begin
    if os.path.exists(os.path.join(base_dir, 'checkpoint_epoch{}.pth.tar'.format(epoch_num-1))):
        exit(0)

    save_point_file = os.path.join(base_dir, 'model_best.pth.tar')
    if os.path.exists(save_point_file):
        checkpoint_params = torch.load(save_point_file)

        start_epoch = checkpoint_params['epoch']

        best_vali_metric = checkpoint_params['best_vali_metric']

        model.load_state_dict(checkpoint_params['model'])
        optimizer.load_state_dict(checkpoint_params['optimizer'])

        print('checkpoint detected, epoch: %d' % start_epoch)
        stdout.flush()

    # start training
    len_data = len(train_dataloader)

    for epoch in range(start_epoch, epoch_num):
        print("------------epoch {}---------------".format(epoch))

        # scheduler.step(val_dc)
        # print("learning rate = {}".format(optimizer.param_groups[0]['lr']))
        # stdout.flush()
        train_losses = AverageMeter()  # total = idv, sep
        train_metrics = AverageMeter()  # ov, nparc

        for i, (feats, out_feat, label, _) in enumerate(tqdm(train_dataloader)):
            B = label.shape[0]
            evals, losses = train_one_step(feats, out_feat)

            train_losses.update(losses, n=B)
            train_metrics.update(evals, n=B)

        train_losses_avg = train_losses.avg
        train_metrics_avg = train_metrics.avg

        print('<<<Train>>>  Loss: {loss:.4f} || PearsonR: {pearsonr:.4f}'.format(
            loss=train_losses_avg,
            pearsonr=train_metrics_avg
        ))

        # validation
        val_losses_avg, val_metrics_avg = val_during_training(val_dataloader)
        print('<Validation> Loss: {loss:.4f} || PearsonR: {pearsonr:.4f}'.format(
            loss=val_losses_avg,
            pearsonr=val_metrics_avg
        ))

        # save check point & best model in each epoch
        is_best = val_metrics_avg > best_vali_metric  ####  !!!! IMPORTANT !!!!!
        best_vali_metric = max(val_metrics_avg, best_vali_metric)  ####  !!!! IMPORTANT !!!!!

        state = {
            'epoch': epoch + 1,

            'best_vali_metric': best_vali_metric,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        if epoch >= save_epoch_step and epoch % save_epoch_step == 0:
            torch.save(state, os.path.join(base_dir, 'checkpoint_epoch{}.pth.tar'.format(epoch)))
        if is_best:
            torch.save(state, os.path.join(base_dir, 'model_best.pth.tar'))

        report_gpu()

    print("=== model end training ===")
    stdout.flush()
