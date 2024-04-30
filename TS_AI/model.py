#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023/06/25

@author: Li Chengyi, https://github.com/li-chengyi

Contact: lichengyi_bit@126.com

"""

import torch
import numpy as np
import torch.nn as nn

from sphericalunet.utils.utils import Get_neighs_order, Get_upconv_index, get_upsample_order
from sphericalunet.models.layers import onering_conv_layer_batch, pool_layer_batch, upconv_layer_batch
from sphericalunet.models.layers import onering_conv_layer, pool_layer, upconv_layer, upsample_interpolation

from einops import rearrange, reduce

"""batch"""
# class general_conv_mesh(nn.Module):
#     def __init__(self, in_ch, out_ch, neigh_orders, norm='bn', act_type='lrelu'):
#         super(general_conv_mesh, self).__init__()
#
#         self.conv = onering_conv_layer_batch(in_ch, out_ch, neigh_orders)
#
#         if norm == 'bn':
#             self.norm = nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False)
#         elif norm == 'gn':
#             self.norm = nn.GroupNorm(4, out_ch)
#         elif norm == 'in':
#             self.norm = nn.InstanceNorm1d(out_ch)
#         elif norm == 'sync_bn':
#             self.norm = SynchronizedBatchNorm1d(out_ch)
#         else:
#             raise ValueError('normalization type {} is not supported'.format(norm))
#
#         if act_type == 'relu':
#             self.activation = nn.ReLU(inplace=True)
#         elif act_type == 'lrelu':
#             self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         else:
#             raise ValueError('activation type {} is not supported'.format(act_type))
#
#
#     def forward(self, x):
#         """
#
#         :param x: (...,V,C)
#         :return: (...,V,C)
#         """
#         # print('general_conv_mesh: x ', x.shape)
#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.activation(x)
#         return x


"""single"""


# class general_conv_mesh(nn.Module):
#     def __init__(self, in_ch, out_ch, neigh_orders, norm='bn', act_type='lrelu'):
#         super(general_conv_mesh, self).__init__()
#
#         self.conv = onering_conv_layer(in_ch, out_ch, neigh_orders)
#
#         if norm == 'bn':
#             self.norm = nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False)
#         elif norm == 'gn':
#             self.norm = nn.GroupNorm(4, out_ch)
#         elif norm == 'in':
#             self.norm = nn.InstanceNorm1d(out_ch)
#         elif norm == 'sync_bn':
#             self.norm = SynchronizedBatchNorm1d(out_ch)
#         else:
#             self.norm = nn.Identity()
#             # raise ValueError('normalization type {} is not supported'.format(norm))
#
#         if act_type == 'relu':
#             self.activation = nn.ReLU(inplace=True)
#         elif act_type == 'lrelu':
#             self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         else:
#             self.activation = nn.Identity()
#             # raise ValueError('activation type {} is not supported'.format(act_type))
#
#     def forward(self, x):
#         """
#         :param x: (B,C,V)
#         :return:
#         """
#         # batch norm version
#         y = []
#         for b in range(x.shape[0]):
#             y1 = self.conv(x[b, ...].permute(1, 0))
#             y1 = self.norm(y1)
#             y.append(self.activation(y1).permute(1, 0))
#         y = torch.stack(y, dim=0)  # ( B, C, V)
#         return y


"""v1.0"""
# class down_block(nn.Module):
#     """
#     downsampling block in spherical unet
#     residual connection
#     """
#
#     def __init__(self, in_ch, out_ch, neigh_orders, pool_neigh_orders, first=False):
#         super(down_block, self).__init__()
#         conv_layer = onering_conv_layer
#
#         #        Batch norm version
#         if first:
#             self.conv1 = nn.Sequential(
#                 conv_layer(in_ch, out_ch, neigh_orders),
#                 nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#                 nn.LeakyReLU(0.2, inplace=True),
#             )
#         else:
#             self.conv1 = nn.Sequential(
#                 pool_layer(pool_neigh_orders, 'mean'),
#                 conv_layer(in_ch, out_ch, neigh_orders),
#                 nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#                 nn.LeakyReLU(0.2, inplace=True),
#             )
#
#         self.conv2 = nn.Sequential(
#             conv_layer(out_ch, out_ch, neigh_orders),
#             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.conv3 = nn.Sequential(
#             conv_layer(out_ch, out_ch, neigh_orders),
#             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#     def forward(self, x):
#         y = []
#         for b in range(x.shape[0]):
#             y1 = x[b,...].permute(1,0)
#             y1 = self.conv1(y1)
#             y1 = y1 + self.conv3(self.conv2(y1))
#             y.append(y1.permute(1,0))
#         y = torch.stack(y, dim=0)  # ( B, C, V)
#         return y


# class up_block(nn.Module):
#     """Define the upsamping block in spherica uent
#     upconv => (conv => BN => ReLU) * 2
#
#     Parameters:
#             in_ch (int) - - input features/channels
#             out_ch (int) - - output features/channels
#             neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders
#
#     """
#
#     def __init__(self, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
#         super(up_block, self).__init__()
#
#         self.up = upconv_layer_batch(in_ch, out_ch, upconv_top_index, upconv_down_index)
#         self.conv = general_conv_mesh(2 * out_ch, out_ch, neigh_orders)
#         self.out_proj = nn.Conv1d(out_ch, out_ch, kernel_size=1)
#
#     def forward(self, x1, x2):
#         """
#         :param x1: (B,C,V)
#         :param x2: (B,C,V)
#         :return:
#         """
#         x1 = self.up(x1)
#         # print('up_block: ', x1.shape, x2.shape)
#         x = torch.cat((x1, x2), 1)  # (B,2*C,V)
#         x = self.out_proj(self.conv(x))  # (B,Co,V)
#         return x


"""
 batch block v2.0
"""
# class down_block(nn.Module):
#     """
#     Batch norm version of
#     downsampling block in spherical transformer unet
#     mean pooling => (conv => BN => ReLU) * 2
#
#     """
#
#     def __init__(self, in_ch, out_ch, neigh_orders, pool_neigh_orders, first=False):
#         super(down_block, self).__init__()
#
#         conv_layer = onering_conv_layer_batch
#         if first:
#             self.block = nn.Sequential(
#                 conv_layer(in_ch, out_ch, neigh_orders),
#                 nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#                 nn.LeakyReLU(0.2, inplace=False),
#                 conv_layer(out_ch, out_ch, neigh_orders),
#                 nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#                 nn.LeakyReLU(0.2, inplace=False)
#             )
#
#
#         else:
#             self.block = nn.Sequential(
#                 pool_layer_batch(pool_neigh_orders),
#                 conv_layer(in_ch, out_ch, neigh_orders),
#                 nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#                 nn.LeakyReLU(0.2, inplace=False),
#                 conv_layer(out_ch, out_ch, neigh_orders),
#                 nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#                 nn.LeakyReLU(0.2, inplace=False),
#             )
#
#     def forward(self, x):
#         # batch norm version
#         x = self.block(x)
#         return x
#
#
# class up_block(nn.Module):
#     """Define the upsamping block in spherica uent
#     upconv => (conv => BN => ReLU) * 2
#
#     Parameters:
#             in_ch (int) - - input features/channels
#             out_ch (int) - - output features/channels
#             neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders
#
#     """
#
#     def __init__(self, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
#         super(up_block, self).__init__()
#
#         conv_layer = onering_conv_layer_batch
#
#         self.up = upconv_layer_batch(in_ch, out_ch, upconv_top_index, upconv_down_index)
#
#         # batch norm version
#         self.double_conv = nn.Sequential(
#             conv_layer(2 * out_ch, out_ch, neigh_orders),
#             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             conv_layer(out_ch, out_ch, neigh_orders),
#             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x = torch.cat((x1, x2), 1)
#         x = self.double_conv(x)
#
#         return x


"""
single
"""


class down_block(nn.Module):
    """
    Batch norm version of
    downsampling block in spherical transformer unet
    mean pooling => (conv => BN => ReLU) * 2

    """

    def __init__(self, in_ch, out_ch, neigh_orders, pool_neigh_orders, first=False):
        super(down_block, self).__init__()

        conv_layer = onering_conv_layer
        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=False),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=False)
            )


        else:
            self.block = nn.Sequential(
                pool_layer(pool_neigh_orders, 'mean'),
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=False),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=False),
            )

    def forward(self, x):
        """
        :param x: (B,C,V)
        :return:
        """
        # batch norm version
        y = []
        for b in range(x.shape[0]):
            y.append(self.block(x[b, ...].permute(1, 0)).permute(1, 0))
        y = torch.stack(y, dim=0)  # ( B, C, V)
        return y


class up_block(nn.Module):
    """Define the upsamping block in spherica uent
    upconv => (conv => BN => ReLU) * 2

    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
            neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders

    """

    def __init__(self, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(up_block, self).__init__()

        conv_layer = onering_conv_layer

        self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)

        # batch norm version
        self.double_conv = nn.Sequential(
            conv_layer(2 * out_ch, out_ch, neigh_orders),
            nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            conv_layer(out_ch, out_ch, neigh_orders),
            nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x1, x2):
        """
        :param x1: (B,C,V)
        :param x2: (B,C,V)
        :return:  (B,C,V)
        """
        y = []
        for b in range(x1.shape[0]):
            y1 = self.up(x1[b, ...].permute(1, 0))  # (V,C)
            y2 = torch.cat((y1, x2[b, ...].permute(1, 0)), 1)  # (V,2C)
            y.append(self.double_conv(y2).permute(1, 0))  # (V,C)
        y = torch.stack(y, dim=0)  # (B,C,V)
        return y


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):
    def __init__(self, chs, neigh_orders):
        super(Encoder, self).__init__()

        self.down1 = down_block(chs[0], chs[1], neigh_orders[0], None, True)
        self.down2 = down_block(chs[1], chs[2], neigh_orders[1], neigh_orders[0])
        self.down3 = down_block(chs[2], chs[3], neigh_orders[2], neigh_orders[1])
        self.down4 = down_block(chs[3], chs[4], neigh_orders[3], neigh_orders[2])
        # self.down5 = down_block(chs[4], chs[5], neigh_orders[4], neigh_orders[3])

    def forward(self, x):
        """
        :param x: (B, V, C_i)
        :return:
        """
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        # x5 = self.down5(x4)
        # print('Encoder: ', x1.shape, x2.shape, x3.shape, x4.shape)
        return x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self, chs, neigh_orders, upconv_down_index, out_ch):
        super(Decoder, self).__init__()

        _, _, \
        upconv_top_index_40962, upconv_down_index_40962, \
        upconv_top_index_10242, upconv_down_index_10242, \
        upconv_top_index_2562, upconv_down_index_2562, \
        upconv_top_index_642, upconv_down_index_642, \
        upconv_top_index_162, upconv_down_index_162 = upconv_down_index

        # self.up4 = up_block(chs[5], chs[4], neigh_orders[3], upconv_top_index_642, upconv_down_index_642)
        self.up3 = up_block(chs[4], chs[3], neigh_orders[2], upconv_top_index_2562, upconv_down_index_2562)
        self.up2 = up_block(chs[3], chs[2], neigh_orders[1], upconv_top_index_10242, upconv_down_index_10242)
        self.up1 = up_block(chs[2], chs[1], neigh_orders[0], upconv_top_index_40962, upconv_down_index_40962)

        self.seg_layer = nn.Conv1d(chs[1], out_ch, kernel_size=1)

    def forward(self, x1, x2, x3, x4):
        """
        :param x1: (B,C,V)
        :param x2: (B,C,V)
        :param x3: (B,C,V)
        :param x4: (B,C,V)
        :return: (B,K,V)
        """
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)  # 40962 * 32

        x = self.seg_layer(x)  # 40962 * K
        return x


class Unet_indiv(nn.Module):
    def __init__(self, modals, in_chs, num_cls, istrain=True):
        super(Unet_indiv, self).__init__()

        neigh_orders = Get_neighs_order()
        neigh_orders = neigh_orders[1:]  # [(286734,), (71694,), (17934,), (4494,), (1134,), (294,), (84,)]

        upconv_indices = Get_upconv_index()

        self.chs = [64, 128, 256, 256]

        self.istrain = istrain

        self.modals = modals
        self.mod_num = len(modals)

        self.idn = nn.Identity()
        self.modal_encoder = Encoder([sum(in_chs[mod] for mod in modals)] + self.chs, neigh_orders)

        self.decoder = Decoder([-1] + self.chs, neigh_orders, upconv_indices, num_cls)
        self.softmax = nn.Softmax(dim=1)
        self.alpha = 0.5

        self.apply(weight_init)

    def forward(self, x):
        """
        :param x: [(B, Ci, V)]i=1:M
        :return: pred: (B, K, V),
                 pred_sofm: (B,K)
        """
        xs = torch.cat(x, dim=1).requires_grad_(True)

        t1, t2, t3, t4 = self.modal_encoder(xs)
        pred = self.decoder(t1, t2, t3, t4)  # (B, K, V)

        pred1 = self.softmax(pred)  # (B, K, V)
        # pred_sum = torch.sum(pred1 * (pred1 > self.alpha), dim=2)  # (B, K)
        pred_sum = pred1.sum(2, keepdim=True) + 1e-3  # (B, K, 1)

        recon = []
        for i, modal in enumerate(self.modals):
            fess1 = torch.einsum('b k v, b c v -> b k c', pred1 / pred_sum, x[i])
            recon.append(torch.einsum('b k v, b k c -> b c v', pred1, fess1))

        if self.istrain:
            return pred, recon
        else:
            return pred, recon, pred_sum[:,:,0], xs


class Unet_syn(nn.Module):
    def __init__(self, modals, out_modal, in_chs):
        super(Unet_syn, self).__init__()

        neigh_orders = Get_neighs_order()
        neigh_orders = neigh_orders[1:]  # [(286734,), (71694,), (17934,), (4494,), (1134,), (294,), (84,)]

        upconv_indices = Get_upconv_index()

        # self.chs = [32, 64, 128, 128]
        self.chs = [64, 128, 256, 256]

        self.modals = modals
        self.mod_num = len(modals)
        self.out_modal = out_modal

        self.modal_encoder = Encoder([sum(in_chs[mod] for mod in modals)] + self.chs, neigh_orders)
        self.recon_decoder = Decoder([-1] + self.chs, neigh_orders, upconv_indices, in_chs[out_modal])

        self.apply(weight_init)

    def forward(self, x):
        """
        :param x: [(B, Ci, V)]i=1:2
        :return: recon: (B, Ci, V)
        """
        t1, t2, t3, t4 = self.modal_encoder(torch.cat(x, dim=1))
        recon = self.recon_decoder(t1, t2, t3, t4)

        return recon


class Unet_2branch(nn.Module):
    def __init__(self, modals, out_modal, in_chs, parcel_cls, istrain=True):
        super(Unet_2branch, self).__init__()

        neigh_orders = Get_neighs_order()
        neigh_orders = neigh_orders[1:]  # [(286734,), (71694,), (17934,), (4494,), (1134,), (294,), (84,)]

        upconv_indices = Get_upconv_index()

        self.chs = [64, 128, 256, 256]

        self.istrain = istrain

        self.modals = modals
        self.mod_num = len(modals)

        self.idn = nn.Identity()
        self.modal_encoder = Encoder([sum(in_chs[mod] for mod in modals)] + self.chs, neigh_orders)

        self.syn_decoder = Decoder([-1] + self.chs, neigh_orders, upconv_indices, in_chs[out_modal])

        self.indiv_decoder = Decoder([-1] + self.chs, neigh_orders, upconv_indices, parcel_cls)

        self.softmax = nn.Softmax(dim=1)
        self.alpha = 0.5

        self.apply(weight_init)

    def forward(self, x):
        """
        :param x: [(B, Ci, V)]i=1:M
        :return: pred: (B, K, V),
                 pred_sofm: (B,K)
        """
        xs = torch.cat(x, dim=1).requires_grad_(True)

        t1, t2, t3, t4 = self.modal_encoder(xs)

        # syn branch
        syn = self.syn_decoder(t1, t2, t3, t4)

        # indiv branch
        pred = self.indiv_decoder(t1, t2, t3, t4)  # (B, K, V)
        pred1 = self.softmax(pred)  # (B, K, V)
        # pred_sum = torch.sum(pred1 * (pred1 > self.alpha), dim=2)  # (B, K)
        pred_sum = pred1.sum(2, keepdim=True) + 1e-3  # (B, K, 1)

        feat_parc_avg = []
        for i, modal in enumerate(self.modals):
            fess1 = torch.einsum('b k v, b c v -> b k c', pred1 / pred_sum, x[i])
            feat_parc_avg.append(torch.einsum('b k v, b k c -> b c v', pred1, fess1))

        # task also feat
        fess1 = torch.einsum('b k v, b c v -> b k c', pred1 / pred_sum, syn)
        feat_parc_avg.append(torch.einsum('b k v, b k c -> b c v', pred1, fess1))

        if self.istrain:
            return syn, pred, feat_parc_avg
        else:
            return syn, pred, feat_parc_avg, pred_sum[:,:,0], xs


from torch.cuda.amp import autocast, GradScaler

# if __name__ == '__main__':
#     scaler = torch.cuda.amp.GradScaler()
#     # from torch.cuda.amp import autocast as autocast
#     print(torch.__version__)
#     print(torch.version.cuda)
#     print(torch.cuda.amp)
#
#     device = 'cuda'
#
#     modals = ['r', 't']
#     in_chs = {'r': 50, 't': 47}
#     out_ch = 180
#     B, V = 8, 40962
#
#     model = RFNet(modals, in_chs, out_ch).to(device)
#
#     masks = []
#     for b in range(B):
#         while True:
#             mask = torch.randint(2, (2,)) > 0.5
#             if mask.sum() > 0:
#                 masks.append(mask)
#                 break
#     masks = torch.stack(masks, dim=0)
#
#     with autocast():
#         idv, sep = model([torch.randn(B, C_i, V).float().to(device) for C_i in [50, 47]], masks.to(device))
#
#     print(idv.shape)
#     print([ss.shape for ss in sep])
