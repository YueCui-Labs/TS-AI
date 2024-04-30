import os, subprocess
import torch
import torch.nn as nn
import torchvision
import numpy as np

from nilearn import surface
import nibabel as nib

from sphericalunet.utils.vtk import read_vtk

import scipy.io as scio
from scipy.spatial.ckdtree import cKDTree
# from transformations import transformations

import random


def coord_normalize(coord, length):
    return coord / np.sqrt(np.sum(coord ** 2, axis=1, keepdims=True)) * length  # normalize


def feat_normalize(feat, label):
    """
    :param feat: (C,V)
    :param label: (V)
    :return:
    """
    mu = feat[:, label > 0].mean(1, keepdim=True)
    sigma = feat[:, label > 0].std(1, keepdim=True)
    feat[:, label > 0] = (feat[:, label > 0] - mu) / (sigma + 1e-4)
    return feat  # normalize


class unet_syn_dataset(torch.utils.data.Dataset):
    def __init__(self, sub_list_file, modals, out_modal, feat_dir, label_40k_file, hemi,
                 transform=True, istrain=True, r_max=32, n_warp=100):
        self.sub_list = np.loadtxt(sub_list_file, dtype=str)
        self.feat_dir = feat_dir
        self.hemi = hemi

        self.transform = transform
        vert_gen = scio.loadmat('neigh_indices/vertices_genertation_pair_40962.mat')
        self.vert_gen = vert_gen['vert_gen_pair'] - 1  # (40962, 2)
        self.od = [12, 42, 162, 642, 2562, 10242, 40962]
        sphere_40k = read_vtk('neigh_indices/Sphere.40k.vtk')
        self.vert_coord = sphere_40k['vertices']
        self.r_max = r_max
        self.n_warp = n_warp

        self.modals = modals
        print(self.modals)
        self.num_modal = len(modals)
        self.out_modal = out_modal

        self.normalize = True

        label = load_all_vert_gii(label_40k_file)  # (#vert,)
        lookup = np.unique(label[label > 0])
        self.label = label_continue(label, lookup)
        # print(lookup)

        self.modal_name = {
            'r': 'idv_scan1_gig_ICd100_47_MSMAll',
            't': 'task_contrast24_MSMAll',
            'd': 'dti_tract41_MSMAll',
            'g': 'geo_modes50_MSMAll',
            # 'd': 'dti_tract41',
            # 'g': 'geo_modes50',
        }

        self.istrain=istrain

    def __getitem__(self, index):
        sub = self.sub_list[index]

        feats = []
        for modal in self.modals:
            feature_40k = os.path.join(self.feat_dir, sub,
                                       '{}.{}.40k.func.gii'.format(self.modal_name[modal], self.hemi))
            feat = surface.load_surf_data(feature_40k).astype(np.float32)
            feats.append(feat.T if feat.shape[0]==40962 else feat)

            # print(feat.shape, type(feat), )
            if modal == 'r' and np.sum(np.isnan(feat)) > 0:
                print(sub)

        if self.istrain:
            feature_40k = os.path.join(self.feat_dir, sub,
                                       '{}.{}.40k.func.gii'.format(self.modal_name[self.out_modal], self.hemi))
            out_feat = surface.load_surf_data(feature_40k).astype(np.float32)
            out_feat = out_feat.T if out_feat.shape[0] == 40962 else out_feat

        label = self.label

        if self.transform:
            coords = random_warp_coords(self.vert_coord, self.od, self.vert_gen, self.r_max,
                                        self.n_warp)  # nonlinear warping
            # coords = np.matmul(coords, transformations.random_rotation_matrix()[:3, :3])  # random rotation

            _, col = cKDTree(coords).query(self.vert_coord, 1)

            feats = [f[:, col] for f in feats]
            if self.istrain:
                out_feat = out_feat[:, col]
            label = label[col]

        feats = [torch.from_numpy(f).float() for f in feats]
        if self.istrain:
            out_feat = torch.from_numpy(out_feat).float()
        label = torch.from_numpy(label).long()

        if self.normalize:
            feats = [feat_normalize(f, label) for f in feats]
            if self.istrain:
                out_feat = feat_normalize(out_feat, label)

        if self.istrain:
            return tuple(feats), out_feat, label, sub
        else:
            return tuple(feats), label, sub

    def __len__(self):
        return len(self.sub_list)


class unet_indiv_dataset(torch.utils.data.Dataset):
    def __init__(self, sub_list_file, modals, feat_dir, label_40k_file, hemi, syn_task,
                 transform=True, istrain=False, r_max=32, n_warp=100):
        self.sub_list = np.loadtxt(sub_list_file, dtype=str)
        self.feat_dir = feat_dir
        self.hemi = hemi

        self.transform = transform
        vert_gen = scio.loadmat('neigh_indices/vertices_genertation_pair_40962.mat')
        self.vert_gen = vert_gen['vert_gen_pair'] - 1  # (40962, 2)
        self.od = [12, 42, 162, 642, 2562, 10242, 40962]
        sphere_40k = read_vtk('neigh_indices/Sphere.40k.vtk')
        self.vert_coord = sphere_40k['vertices']
        self.r_max = r_max
        self.n_warp = n_warp

        self.modals = modals
        print(self.modals)
        self.num_modal = len(modals)

        self.normalize = True

        label = load_all_vert_gii(label_40k_file)  # (#vert,)
        lookup = np.unique(label[label > 0])
        self.label = label_continue(label, lookup)
        # print(lookup)

        self.modal_name = {
            'r': 'idv_scan1_gig_ICd100_47_MSMAll',
            't': syn_task,
            'd': 'dti_tract41_MSMAll',
            'g': 'geo_modes50_MSMAll',
            # 'd': 'dti_tract41',
            # 'g': 'geo_modes50',
        }

        self.istrain = istrain

    def __getitem__(self, index):
        sub = self.sub_list[index]

        feats = []
        for modal in self.modals:
            feature_40k = os.path.join(self.feat_dir, sub,
                                       '{}.{}.40k.func.gii'.format(self.modal_name[modal], self.hemi))
            feat = surface.load_surf_data(feature_40k).astype(np.float32)
            feats.append(feat.T if feat.shape[0]==40962 else feat)

        label = self.label

        if self.transform:
            coords = random_warp_coords(self.vert_coord, self.od, self.vert_gen, self.r_max,
                                        self.n_warp)  # nonlinear warping
            # coords = np.matmul(coords, transformations.random_rotation_matrix()[:3, :3])  # random rotation

            _, col = cKDTree(coords).query(self.vert_coord, 1)

            feats = [f[:, col] for f in feats]
            label = label[col]

        feats = [torch.from_numpy(f).float() for f in feats]
        label = torch.from_numpy(label).long()

        if self.normalize:
            feats = [feat_normalize(f, label) for f in feats]

        return tuple(feats), label, sub

    def __len__(self):
        return len(self.sub_list)


class unet_syn_dataset_subn(torch.utils.data.Dataset):
    def __init__(self, sub_list_file, modals, out_modal, feat_dir, label_40k_file, hemi,
                 transform=True, istrain=True, r_max=32, n_warp=100, sub_num=388):
        self.sub_list = np.loadtxt(sub_list_file, dtype=str)
        self.sub_num_whole = sub_num  ## TODO: dead write
        self.ttms = len(self.sub_list) // sub_num  ## TODO: dead write

        self.feat_dir = feat_dir
        self.hemi = hemi

        self.transform = transform
        vert_gen = scio.loadmat('neigh_indices/vertices_genertation_pair_40962.mat')
        self.vert_gen = vert_gen['vert_gen_pair'] - 1  # (40962, 2)
        self.od = [12, 42, 162, 642, 2562, 10242, 40962]
        sphere_40k = read_vtk('neigh_indices/Sphere.40k.vtk')
        self.vert_coord = sphere_40k['vertices']
        self.r_max = r_max
        self.n_warp = n_warp

        self.modals = modals
        print(self.modals)
        self.num_modal = len(modals)
        self.out_modal = out_modal

        self.normalize = True

        label = load_all_vert_gii(label_40k_file)  # (#vert,)
        lookup = np.unique(label[label > 0])
        self.label = label_continue(label, lookup)
        # print(lookup)

        self.modal_name = {
            'r': 'idv_scan1_gig_ICd100_47_MSMAll',
            't': 'task_contrast24_MSMAll',
            'd': 'dti_tract41_MSMAll',
            'g': 'geo_modes50_MSMAll',
            # 'd': 'dti_tract41',
            # 'g': 'geo_modes50',
        }

        self.istrain=istrain

    def __getitem__(self, index):
        sub = self.sub_list[index % self.sub_num_whole]

        feats = []
        for modal in self.modals:
            feature_40k = os.path.join(self.feat_dir, sub,
                                       '{}.{}.40k.func.gii'.format(self.modal_name[modal], self.hemi))
            feat = surface.load_surf_data(feature_40k).astype(np.float32)
            feats.append(feat.T if feat.shape[0]==40962 else feat)

            # print(feat.shape, type(feat), )
            if modal == 'r' and np.sum(np.isnan(feat)) > 0:
                print(sub)

        if self.istrain:
            feature_40k = os.path.join(self.feat_dir, sub,
                                       '{}.{}.40k.func.gii'.format(self.modal_name[self.out_modal], self.hemi))
            out_feat = surface.load_surf_data(feature_40k).astype(np.float32)
            out_feat = out_feat.T if out_feat.shape[0] == 40962 else out_feat

        label = self.label

        if self.transform:
            coords = random_warp_coords(self.vert_coord, self.od, self.vert_gen, self.r_max,
                                        self.n_warp)  # nonlinear warping
            # coords = np.matmul(coords, transformations.random_rotation_matrix()[:3, :3])  # random rotation

            _, col = cKDTree(coords).query(self.vert_coord, 1)

            feats = [f[:, col] for f in feats]
            if self.istrain:
                out_feat = out_feat[:, col]
            label = label[col]

        feats = [torch.from_numpy(f).float() for f in feats]
        if self.istrain:
            out_feat = torch.from_numpy(out_feat).float()
        label = torch.from_numpy(label).long()

        if self.normalize:
            feats = [feat_normalize(f, label) for f in feats]
            if self.istrain:
                out_feat = feat_normalize(out_feat, label)

        if self.istrain:
            return tuple(feats), out_feat, label, sub
        else:
            return tuple(feats), label, sub

    def __len__(self):
        return len(self.sub_list)


class unet_indiv_dataset_subn(torch.utils.data.Dataset):
    def __init__(self, sub_list_file, modals, feat_dir, label_40k_file, hemi, syn_task,
                 transform=True, istrain=False, r_max=32, n_warp=100, sub_num=388):
        self.sub_list = np.loadtxt(sub_list_file, dtype=str)
        self.sub_num_whole = sub_num  ## TODO: dead write

        self.feat_dir = feat_dir
        self.hemi = hemi

        self.transform = transform
        vert_gen = scio.loadmat('neigh_indices/vertices_genertation_pair_40962.mat')
        self.vert_gen = vert_gen['vert_gen_pair'] - 1  # (40962, 2)
        self.od = [12, 42, 162, 642, 2562, 10242, 40962]
        sphere_40k = read_vtk('neigh_indices/Sphere.40k.vtk')
        self.vert_coord = sphere_40k['vertices']
        self.r_max = r_max
        self.n_warp = n_warp

        self.modals = modals
        print(self.modals)
        self.num_modal = len(modals)

        self.normalize = True

        label = load_all_vert_gii(label_40k_file)  # (#vert,)
        lookup = np.unique(label[label > 0])
        self.label = label_continue(label, lookup)
        # print(lookup)

        self.modal_name = {
            'r': 'idv_scan1_gig_ICd100_47_MSMAll',
            't': syn_task,
            'd': 'dti_tract41_MSMAll',
            'g': 'geo_modes50_MSMAll',
            # 'd': 'dti_tract41',
            # 'g': 'geo_modes50',
        }

        self.istrain = istrain

    def __getitem__(self, index):
        sub = self.sub_list[index % self.sub_num_whole]

        feats = []
        for modal in self.modals:
            feature_40k = os.path.join(self.feat_dir, sub,
                                       '{}.{}.40k.func.gii'.format(self.modal_name[modal], self.hemi))
            feat = surface.load_surf_data(feature_40k).astype(np.float32)
            feats.append(feat.T if feat.shape[0]==40962 else feat)

        label = self.label

        if self.transform:
            coords = random_warp_coords(self.vert_coord, self.od, self.vert_gen, self.r_max,
                                        self.n_warp)  # nonlinear warping
            # coords = np.matmul(coords, transformations.random_rotation_matrix()[:3, :3])  # random rotation

            _, col = cKDTree(coords).query(self.vert_coord, 1)

            feats = [f[:, col] for f in feats]
            label = label[col]

        feats = [torch.from_numpy(f).float() for f in feats]
        label = torch.from_numpy(label).long()

        if self.normalize:
            feats = [feat_normalize(f, label) for f in feats]

        return tuple(feats), label, sub

    def __len__(self):
        return len(self.sub_list)


class unet_syn_indiv_dataset(torch.utils.data.Dataset):
    def __init__(self, sub_list_file, modals, out_modal, feat_dir, label_40k_file, hemi,
                 transform=True,r_max=32, n_warp=100):
        self.sub_list = np.loadtxt(sub_list_file, dtype=str)
        self.feat_dir = feat_dir
        self.hemi = hemi

        self.transform = transform
        vert_gen = scio.loadmat('neigh_indices/vertices_genertation_pair_40962.mat')
        self.vert_gen = vert_gen['vert_gen_pair'] - 1  # (40962, 2)
        self.od = [12, 42, 162, 642, 2562, 10242, 40962]
        sphere_40k = read_vtk('neigh_indices/Sphere.40k.vtk')
        self.vert_coord = sphere_40k['vertices']
        self.r_max = r_max
        self.n_warp = n_warp

        self.modals = modals
        print(self.modals)
        self.num_modal = len(modals)

        self.normalize = True

        label = load_all_vert_gii(label_40k_file)  # (#vert,)
        lookup = np.unique(label[label > 0])
        self.label = label_continue(label, lookup)
        # print(lookup)

        self.modal_name = {
            'r': 'idv_scan1_gig_ICd100_47_MSMAll',
            't': 'task_contrast24_MSMAll',
            'd': 'dti_tract41_MSMAll',
            'g': 'geo_modes50_MSMAll',
            # 'd': 'dti_tract41',
            # 'g': 'geo_modes50',
        }
        self.out_modal = out_modal

    def __getitem__(self, index):
        sub = self.sub_list[index]

        feats = []
        for modal in self.modals:
            feature_40k = os.path.join(self.feat_dir, sub,
                                       '{}.{}.40k.func.gii'.format(self.modal_name[modal], self.hemi))
            feat = surface.load_surf_data(feature_40k).astype(np.float32)
            feats.append(feat.T if feat.shape[0]==40962 else feat)

        feature_40k = os.path.join(self.feat_dir, sub,
                                   '{}.{}.40k.func.gii'.format(self.modal_name[self.out_modal], self.hemi))
        out_feat = surface.load_surf_data(feature_40k).astype(np.float32)
        out_feat = out_feat.T if out_feat.shape[0] == 40962 else out_feat

        label = self.label

        if self.transform:
            coords = random_warp_coords(self.vert_coord, self.od, self.vert_gen, self.r_max,
                                        self.n_warp)  # nonlinear warping
            # coords = np.matmul(coords, transformations.random_rotation_matrix()[:3, :3])  # random rotation

            _, col = cKDTree(coords).query(self.vert_coord, 1)

            feats = [f[:, col] for f in feats]
            out_feat = out_feat[:, col]

            label = label[col]

        feats = [torch.from_numpy(f).float() for f in feats]
        out_feat = torch.from_numpy(out_feat).float()
        label = torch.from_numpy(label).long()

        if self.normalize:
            feats = [feat_normalize(f, label) for f in feats]
            out_feat = feat_normalize(out_feat, label)

        return tuple(feats), out_feat, label, sub

    def __len__(self):
        return len(self.sub_list)


def random_warp_coords(vert_coord, od, vert_gen, r_max=32, warp_num=100):
    vert_coord = vert_coord.copy()
    warp_ind = np.random.choice(a=od[2], size=warp_num, replace=False, p=None)
    warp_ind.sort()  # (100,)

    # ball volume uniform sampling
    r_ = r_max * np.random.uniform(0, 1, (warp_num, 1)) ** (1 / 3)
    theta_ = np.arccos(1 - 2 * np.random.uniform(0, 1, (warp_num, 1)))
    phi_ = 2 * np.pi * np.random.uniform(0, 1, (warp_num, 1))

    x_ = r_ * np.sin(theta_) * np.cos(phi_)
    y_ = r_ * np.sin(theta_) * np.sin(phi_)
    z_ = r_ * np.cos(theta_)

    ico2_coord_delta = np.concatenate((x_, y_, z_), axis=1)

    # warp ico2 vertices
    vert_coord[warp_ind, :] = vert_coord[warp_ind, :] + ico2_coord_delta
    vert_coord[warp_ind, :] = coord_normalize(vert_coord[warp_ind, :], 100)

    # regenerate ico3,4,5,6 vertices by Triangulation
    for i in range(2, 6):
        od_i, od_i1 = od[i], od[i + 1]
        ico_i_gen = vert_gen[od_i:od_i1, :]
        vert_coord[od_i:od_i1, :] = (vert_coord[ico_i_gen[:, 0], :] + vert_coord[ico_i_gen[:, 1], :]) / 2
        vert_coord[od_i:od_i1, :] = coord_normalize(vert_coord[od_i:od_i1, :], 100)

    return vert_coord


def ball_sampling(r_max, warp_num):
    r_ = r_max * np.random.uniform(0, 1, (warp_num, 1)) ** (1 / 3)
    theta_ = np.arccos(1 - 2 * np.random.uniform(0, 1, (warp_num, 1)))
    phi_ = 2 * np.pi * np.random.uniform(0, 1, (warp_num, 1))

    x_ = r_ * np.sin(theta_) * np.cos(phi_)
    y_ = r_ * np.sin(theta_) * np.sin(phi_)
    z_ = r_ * np.cos(theta_)
    return x_, y_, z_


def random_warp_coords2(vert_coord, od, vert_gen, r_max=32, warp_num=100):
    vert_coord = vert_coord.copy()

    ### ico2
    warp_ind = np.random.choice(a=od[2], size=warp_num, replace=False, p=None)
    warp_ind.sort()  # (100,)

    # ball volume uniform sampling
    ico2_coord_delta = np.concatenate((ball_sampling(r_max, warp_num)), axis=1)

    # warp ico2 vertices
    vert_coord[warp_ind, :] = vert_coord[warp_ind, :] + ico2_coord_delta
    vert_coord[warp_ind, :] = coord_normalize(vert_coord[warp_ind, :], 100)

    # regenerate ico4,5,6 vertices by Triangulation
    for i in range(2, 6):
        od_i, od_i1 = od[i], od[i + 1]
        ico_i_gen = vert_gen[od_i:od_i1, :]
        vert_coord[od_i:od_i1, :] = (vert_coord[ico_i_gen[:, 0], :] + vert_coord[ico_i_gen[:, 1], :]) / 2
        vert_coord[od_i:od_i1, :] = coord_normalize(vert_coord[od_i:od_i1, :], 100)

    ### ico3
    warp_ind = np.random.choice(a=od[3], size=warp_num * 4, replace=False, p=None)
    warp_ind.sort()  # (100,)

    # ball volume uniform sampling
    ico3_coord_delta = np.concatenate((ball_sampling(r_max / 2, warp_num * 4)), axis=1)

    # warp ico3 vertices
    vert_coord[warp_ind, :] = vert_coord[warp_ind, :] + ico3_coord_delta
    vert_coord[warp_ind, :] = coord_normalize(vert_coord[warp_ind, :], 100)

    # regenerate ico4,5,6 vertices by Triangulation
    for i in range(3, 6):
        od_i, od_i1 = od[i], od[i + 1]
        ico_i_gen = vert_gen[od_i:od_i1, :]
        vert_coord[od_i:od_i1, :] = (vert_coord[ico_i_gen[:, 0], :] + vert_coord[ico_i_gen[:, 1], :]) / 2
        vert_coord[od_i:od_i1, :] = coord_normalize(vert_coord[od_i:od_i1, :], 100)

    return vert_coord


def load_all_vert_gii(file_path):
    file = nib.load(file_path)
    cdata = file.darrays[0].data.squeeze()
    cdata[cdata < 0] = 0
    return cdata


def label_continue(label, lookup):
    cont = np.copy(label)
    for p, lookup_p in enumerate(lookup):
        cont[label == lookup_p] = p + 1
    return cont


from utils import write_vtk
import vtk

if __name__ == '__main__':
    adj10242 = scio.loadmat('neigh_indices/adj_mat.mat')['adj_mat']
    adj2562 = scio.loadmat('neigh_indices/adj_mat_2562.mat')['adj_mat_2562']
    adj642 = scio.loadmat('neigh_indices/adj_mat_642.mat')['adj_mat_642']

    sphere_40k = read_vtk('neigh_indices/Sphere.40k.vtk')
    vert_coord = sphere_40k['vertices']

    for adj, vnum in zip([adj10242, adj2562, adj642], [10242, 2562, 642]):

        sphere_warped_vtk = os.path.join('neigh_indices/Sphere.{}.vtk'.format(vnum))
        sphere_warped_gii = os.path.join('neigh_indices/Sphere.{}.surf.gii'.format(vnum))

        # write sphere file and map feature/label 32k file to 40k file
        coords = vert_coord[:vnum, :]  # nonlinear warping
        faces = []
        kss = cKDTree(coords)

        for i in range(vnum):
            ddj = np.where(adj[i] == 1)[0]
            ddj = ddj[ddj > i]
            for j in ddj:
                ddk = np.where(np.bitwise_and(adj[i] == 1, adj[j] == 1))[0]
                ddk = ddk[ddk > j]
                for k in ddk:
                    _, col = kss.query(coords[[i, j, k], :], 20)
                    disp = coords[col.flatten(), :].mean(0) - coords[[i, j, k], :].mean(0)
                    norm = np.cross(coords[j, :] - coords[i, :], coords[k, :] - coords[j, :])
                    if norm.dot(disp) > 0:
                        faces.append([i, j, k])
                    else:
                        faces.append([i, k, j])
        # faces = np.stack(faces, dtype=np.int32)

        # write_vtk(coords, faces, sphere_warped_vtk)
        points = vtk.vtkPoints()
        for coord in coords:
            points.InsertNextPoint(coord[0], coord[1], coord[2])

        triangles = vtk.vtkCellArray()
        for face in faces:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, face[0])
            triangle.GetPointIds().SetId(1, face[1])
            triangle.GetPointIds().SetId(2, face[2])
            triangles.InsertNextCell(triangle)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(triangles)

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(sphere_warped_vtk)
        writer.SetInputData(polydata)
        writer.Write()

        p = subprocess.Popen('surf2surf -i {} -o {}'.format(sphere_warped_vtk, sphere_warped_gii),
                             shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return_code = p.wait()
        if return_code != 0:
            exit(1)
