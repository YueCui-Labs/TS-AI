import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange
import math


class ce_const_loss(nn.Module):
    def __init__(self, lam_feat):
        super(ce_const_loss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.l2 = nn.MSELoss()

        self.lam_feat = lam_feat

    def forward(self, fuse, gt, orgin, recon):
        """
        :param fuse: (B, K, V)
        :param gt: (B,V) 0~L
        :param orgin: [(B,Ci,V)]i:1:M
        :param recon: [(B,Ci,V)]i=1:M
        """
        loss_ce = self.ce(fuse.permute(0, 2, 1)[gt > 0, ...], gt[gt > 0].flatten() - 1)
        loss_feat = self.l2(torch.cat(orgin, dim=1), torch.cat(recon, dim=1))

        loss = loss_ce + self.lam_feat * loss_feat
        return loss, loss_ce, loss_feat


class dice_const_loss(nn.Module):
    def __init__(self, lam_std):
        super(dice_const_loss, self).__init__()
        self.l2 = nn.MSELoss()

        self.lam_std = lam_std

    def forward(self, fuse, gt, orgin, recon, eps=1e-5):
        """
        :param fuse: (B, K, V)
        :param gt: (B,V) 0~K
        :param orgin: [(B,Ci,V)]i:1:M
        :param recon: [(B,Ci,V)]i=1:M
        """

        target = torch.nn.functional.one_hot(gt)[:,:,1:].permute(0,2,1)  # (B, K, V) 0~K
        # print(target.shape)
        # predictive = fuse.permute(0, 2, 1)[gt > 0, ...]  # (B, K)
        intersection = 2 * torch.sum(fuse * target, dim=2) + eps  # (B, K)
        union = torch.sum(fuse, dim=2) + torch.sum(target, dim=2) + eps  # (B, K)
        loss_dice = (1 - intersection / union).mean()

        loss_const = self.l2(torch.cat(orgin, dim=1), torch.cat(recon, dim=1))

        loss = loss_dice + self.lam_std * loss_const
        return loss, loss_dice, loss_const


class syn_loss(nn.Module):
    def __init__(self):
        super(syn_loss, self).__init__()
        self.l2 = nn.MSELoss()

    def forward(self, orgin, recon):
        """
        :param orgin: (B,C,V)
        :param recon: (B,C,V)
        :return
        """
        return self.l2(orgin, recon)


class syn_indiv_loss(nn.Module):
    def __init__(self, lam_feat, lam_syn):
        super(syn_indiv_loss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.l2 = nn.MSELoss()

        self.lam_feat = lam_feat
        self.lam_syn = lam_syn

    def forward(self, fuse, gt, orgin_infeat, recon_infeat, orgin_outfeat, recon_outfeat):
        """
        :param fuse: (B, K, V)
        :param gt: (B,V) 0~L
        :param orgin_infeat: [(B,Ci,V)]i:1:M
        :param recon_infeat: [(B,Ci,V)]i=1:M

        :param orgin_outfeat: (B,C,V)
        :param recon_outfeat: (B,C,V)
        :return
        """

        loss_ce = self.ce(fuse.permute(0, 2, 1)[gt > 0, ...], gt[gt > 0].flatten() - 1)
        loss_feat = self.l2(torch.cat([*orgin_infeat, orgin_outfeat], dim=1), torch.cat(recon_infeat, dim=1))
        # task also feat
        # loss_feat = self.l2(torch.cat([*orgin_infeat, orgin_outfeat], dim=1), torch.cat(recon_infeat, dim=1))
        loss_syn = self.l2(orgin_outfeat, recon_outfeat)

        loss = loss_ce + self.lam_feat * loss_feat + self.lam_syn * loss_syn
        return loss, loss_ce, loss_feat, loss_syn

