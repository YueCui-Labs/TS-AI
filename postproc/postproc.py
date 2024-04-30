import numpy as np

from nilearn import surface
import nibabel as nib

from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.sparse import load_npz, save_npz

import os, subprocess


def gen_agj_mat():
    for hemi in ['L', 'R']:
    # for hemi in ['R']:
    # for hemi in ['L']:
        surf = surface.load_surf_data('fsaverage.{}.midthickness.32k_fs_LR.surf.gii'.format(hemi))
        vertices, faces = surf[0], surf[1]
        print(vertices.shape, faces.shape)

        # Create a sparse adjacency matrix
        N = len(vertices)
        adjacency_matrix = np.zeros((N,N), dtype=np.float32)
        for face in faces:
            # Set the entries of the adjacency matrix based on the distances between connected vertices
            for i in range(3):
                for j in range(i + 1, 3):
                    vertex1 = face[i]
                    vertex2 = face[j]
                    distance = np.linalg.norm(vertex1 - vertex2)
                    adjacency_matrix[vertex1, vertex2] = distance
                    adjacency_matrix[vertex2, vertex1] = distance

        adjacency_matrix = csr_matrix(adjacency_matrix)

        # Print the adjacency matrix
        print(adjacency_matrix.toarray())
        save_npz("adjacency_matrix_{}.npz".format(hemi), adjacency_matrix)


def gen_shortest_mat():
    for hemi in ['L', 'R']:
    # for hemi in ['R']:
    # for hemi in ['L']:
        adjacency_matrix = load_npz("adjacency_matrix_{}.npz".format(hemi))

        dist_matrix, predecessors = shortest_path(csgraph=adjacency_matrix, directed=False, return_predecessors=True)
        print(dist_matrix.shape, predecessors.shape)

        np.save('dist_matrix_{}.npy'.format(hemi), dist_matrix)
        np.save('predecessors_{}.npy'.format(hemi), predecessors)


def remove_small_patches():
    # for hemi in ['L', 'R']:
    # for hemi in ['R']:
    for hemi in ['L']:
        areas = surface.load_surf_data('fsaverage.{}.midthickness_areas.32k_fs_LR.func.gii'.format(hemi))
        print(areas.shape)

        labels = surface.load_surf_data('MMLP_Glasser_{}.32k_fs_LR.label.gii'.format(hemi))
        print(labels.shape)

        adjacency_matrix = load_npz("adjacency_matrix_{}.npz".format(hemi))

        # Detect connected components for each label
        labels_new = np.array(labels)
        for label in np.unique(labels)[1:]:
            # Find the indices of nodes with the current label
            label_indices = np.where(labels == label)[0]

            # Create a subgraph with only nodes with the current label
            subgraph = adjacency_matrix[label_indices][:, label_indices]

            # Detect connected components for the subgraph
            n_components, component_labels = connected_components(subgraph)

            # Print the connected components for the current label
            for component in range(n_components):
                component_indices = np.where(component_labels == component)[0]
                component_nodes = label_indices[component_indices]
                if areas[component_nodes].sum() < 25:
                    labels_new[component_labels] = 0

        template = nib.load('MMLP_Glasser_{}.32k_fs_LR.label.gii'.format(hemi))
        template.remove_gifti_data_array(0)
        template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.array(labels_new).astype(np.int32)))
        nib.loadsave.save(template, 'MMLP_Glasser_rms_{}.32k_fs_LR.label.gii'.format(hemi))


def join_nearby_patch():
    # for hemi in ['L', 'R']:
    # for hemi in ['R']:
    for hemi in ['L']:
        distance_matrix = np.load('dist_matrix_{}.npy'.format(hemi))
        predecessors = np.load('predecessors_{}.npy'.format(hemi))
        print(distance_matrix.shape, predecessors.shape)
        print('read file done.')

        labels = surface.load_surf_data('MMLP_Glasser_rms_{}.32k_fs_LR.label.gii'.format(hemi))
        print(labels.shape)

        adjacency_matrix = load_npz("adjacency_matrix_{}.npz".format(hemi))

        labels_new = np.array(labels)
        for label in np.unique(labels)[1:]:
            # Find the indices of nodes with the current label
            label_indices = np.where(labels == label)[0]

            # Create a subgraph with only nodes with the current label
            subgraph = adjacency_matrix[label_indices][:, label_indices]

            # Detect connected components for the subgraph
            n_components, component_labels = connected_components(subgraph)

            # Print the connected components for the current label
            for i in range(n_components):
                ci = label_indices[np.where(component_labels == i)[0]]  # index of component_i
                for j in range(i+1, n_components):
                    cj = label_indices[np.where(component_labels == j)[0]]  # index of component_j
                    # print(ci.shape, cj.shape)
                    cis = np.where(distance_matrix[ci,:][:,cj] < 4)  # source (com_i) and target (com_j) nodes
                    # print(cis)
                    if cis[0].size ==0:
                        break
                    ii, jj = cis
                    src_ind = ci[ii]  # source (com_i) nodes
                    tgt_ind = cj[jj]  # target (com_i) nodes
                    for src, tgt in zip(src_ind, tgt_ind):
                        curr_node = tgt
                        while curr_node != src:
                            labels_new[curr_node] = label
                            curr_node = predecessors[src, curr_node]

        template = nib.load('MMLP_Glasser_rms_{}.32k_fs_LR.label.gii'.format(hemi))
        template.remove_gifti_data_array(0)
        template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.array(labels_new).astype(np.int32)))
        nib.loadsave.save(template, 'MMLP_Glasser_rms_jnp_{}.32k_fs_LR.label.gii'.format(hemi))


def multi_patches_ctrl():
    # for hemi in ['L', 'R']:
    # for hemi in ['R']:
    for hemi in ['L']:
        distance_matrix = np.load('dist_matrix_{}.npy'.format(hemi))
        print(distance_matrix.shape)
        print('read file done.')

        areas = surface.load_surf_data('fsaverage.{}.midthickness_areas.32k_fs_LR.func.gii'.format(hemi))
        print(areas.shape)

        labels = surface.load_surf_data('MMLP_Glasser_rms_jnp_{}.32k_fs_LR.label.gii'.format(hemi))
        print(labels.shape)

        adjacency_matrix = load_npz("adjacency_matrix_{}.npz".format(hemi))

        labels_new = np.array(labels)
        for label in np.unique(labels)[1:]:
            label_indices = np.where(labels == label)[0]
            subgraph = adjacency_matrix[label_indices][:, label_indices]
            n_components, component_labels = connected_components(subgraph)

            if n_components > 1:
                # reomve the patches which the area is smaller than the largest area
                component_areas = []
                component_indices = []
                for i in range(n_components):
                    ci = label_indices[np.where(component_labels == i)[0]]
                    component_indices.append(ci)  # index of component_i
                    component_areas.append(areas[ci].sum())
                max_area = max(component_areas)

                for i in range(n_components):
                    ci = component_indices[i]  # index of component_i
                    if component_areas[i] < max_area/3:
                        labels_new[ci] = 0
        labels = labels_new

        labels_new = np.array(labels)
        for label in np.unique(labels)[1:]:
            label_indices = np.where(labels == label)[0]
            subgraph = adjacency_matrix[label_indices][:, label_indices]
            n_components, component_labels = connected_components(subgraph)

            if n_components > 1:
                # reomve the patches which the distance from its nearest component is larger than 30mm
                component_indices = []
                n_component_nodes = []
                for i in range(n_components):
                    ci = label_indices[np.where(component_labels == i)[0]]
                    component_indices.append(ci)  # index of component_i
                    n_component_nodes.append(len(ci))
                ind = np.argsort(n_component_nodes)
                # print(ind)
                for i in ind[:-1]:
                    ci = component_indices[i]  # index of component_i
                    cj = np.concatenate(component_indices[:i] + component_indices[i+1:])
                    if np.min(distance_matrix[ci,:][:,cj]) > 30:
                        labels_new[ci] = 0

        template = nib.load('MMLP_Glasser_rms_jnp_{}.32k_fs_LR.label.gii'.format(hemi))
        template.remove_gifti_data_array(0)
        template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.array(labels_new).astype(np.int32)))
        nib.loadsave.save(template, 'MMLP_Glasser_rms_jnp_mpc_{}.32k_fs_LR.label.gii'.format(hemi))


def label_dil():
    # for hemi in ['L', 'R']:
    # for hemi in ['R']:
    for hemi in ['L']:
        indiv_label = 'MMLP_Glasser_rms_jnp_mpc_{}.32k_fs_LR.label.gii'.format(hemi)
        group_label = 'fsaverage.{}.Glasser.32k_fs_LR.label.gii'.format(hemi)
        surf = 'fsaverage.{}.midthickness.32k_fs_LR.surf.gii'.format(hemi)
        dilate_dist = 20
        post_label = 'MMLP_Glasser_rms_jnp_mpc_dil_{}.32k_fs_LR.label.gii'.format(hemi)

        p = subprocess.Popen("wb_command -label-dilate {} {} {} {}".format(
            indiv_label, surf, dilate_dist, post_label),
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return_code = p.wait()
        if return_code != 0:
            print('wb_command -label-dilate failed')
            exit(1)

        p = subprocess.Popen("wb_command -label-mask {} {} {}".format(
            post_label, group_label, post_label),
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return_code = p.wait()
        if return_code != 0:
            print('wb_command -label-mask failed')
            exit(1)


def post_proc(sub_list_file, label_dir, method, hemi):
    sub_list = np.loadtxt(sub_list_file, dtype=str)
    print(sub_list.shape)

    # for hemi in ['L', 'R']:
    # for hemi in ['R']:
    # for hemi in ['L']:
    areas = surface.load_surf_data('fsaverage.{}.midthickness_areas.32k_fs_LR.func.gii'.format(hemi))
    adjacency_matrix = load_npz("adjacency_matrix_{}.npz".format(hemi))

    template = nib.load('fsaverage.{}.Glasser.32k_fs_LR.label.gii'.format(hemi))

    distance_matrix = np.load('dist_matrix_{}.npy'.format(hemi))
    predecessors = np.load('predecessors_{}.npy'.format(hemi))
    print(distance_matrix.shape, predecessors.shape)

    if 'Glasser' in method:
        group_label = 'fsaverage.{}.Glasser.32k_fs_LR.label.gii'.format(hemi)
    elif 'BN_Atlas' in method:
        group_label = 'fsaverage.{}.Glasser.32k_fs_LR.label.gii'.format(hemi)
    else:
        print("ERROR: altas not valid, only Glasser or BN_Atlas")
        return
    surf_file = 'fsaverage.{}.midthickness.32k_fs_LR.surf.gii'.format(hemi)

    for ni, sub in enumerate(sub_list):
        label_name = '{}/{}/{}_{}.32k_fs_LR.label.gii'.format(
            label_dir, sub, method, hemi)
        post_label = '{}/{}/{}_post_{}.32k_fs_LR.label.gii'.format(label_dir, sub, method, hemi)
        if not os.path.exists(label_name):
            continue
        if os.path.exists(post_label):
            continue

        labels = surface.load_surf_data(label_name)
        print(ni, sub)

        ## 1. remove small patches
        labels_new = np.array(labels)
        for label in np.unique(labels)[1:]:
            label_indices = np.where(labels == label)[0]
            subgraph = adjacency_matrix[label_indices][:, label_indices]
            n_components, component_labels = connected_components(subgraph)

            for component in range(n_components):
                component_indices = np.where(component_labels == component)[0]
                component_nodes = label_indices[component_indices]
                if areas[component_nodes].sum() < 25:
                    labels_new[component_labels] = 0
        labels = labels_new

        ## 2. join nearby patch
        labels_new = np.array(labels)
        for label in np.unique(labels)[1:]:
            label_indices = np.where(labels == label)[0]
            subgraph = adjacency_matrix[label_indices][:, label_indices]
            n_components, component_labels = connected_components(subgraph)

            for i in range(n_components):
                ci = label_indices[np.where(component_labels == i)[0]]  # index of component_i
                for j in range(i + 1, n_components):
                    cj = label_indices[np.where(component_labels == j)[0]]  # index of component_j
                    cis = np.where(
                        distance_matrix[ci, :][:, cj] < 4)  # source (com_i) and target (com_j) nodes
                    # print(cis)
                    if cis[0].size == 0:
                        break
                    ii, jj = cis
                    src_ind = ci[ii]  # source (com_i) nodes
                    tgt_ind = cj[jj]  # target (com_i) nodes
                    for src, tgt in zip(src_ind, tgt_ind):
                        curr_node = tgt
                        while curr_node != src:
                            labels_new[curr_node] = label
                            curr_node = predecessors[src, curr_node]
        labels = labels_new

        ## 3. reomve the patches which the area is smaller than the 0.33 x largest area
        labels_new = np.array(labels)
        for label in np.unique(labels)[1:]:
            label_indices = np.where(labels == label)[0]
            subgraph = adjacency_matrix[label_indices][:, label_indices]
            n_components, component_labels = connected_components(subgraph)

            if n_components > 1:
                # reomve the patches which the area is smaller than the largest area
                component_areas = []
                component_indices = []
                for i in range(n_components):
                    ci = label_indices[np.where(component_labels == i)[0]]
                    component_indices.append(ci)  # index of component_i
                    component_areas.append(areas[ci].sum())
                max_area = max(component_areas)

                for i in range(n_components):
                    ci = component_indices[i]  # index of component_i
                    if component_areas[i] < max_area / 3:
                        labels_new[ci] = 0
        labels = labels_new

        ## 4. reomve the patches which the distance from its nearest component is larger than 30mm
        labels_new = np.array(labels)
        for label in np.unique(labels)[1:]:
            label_indices = np.where(labels == label)[0]
            subgraph = adjacency_matrix[label_indices][:, label_indices]
            n_components, component_labels = connected_components(subgraph)

            if n_components > 1:
                component_indices = []
                n_component_nodes = []
                for i in range(n_components):
                    ci = label_indices[np.where(component_labels == i)[0]]
                    component_indices.append(ci)  # index of component_i
                    n_component_nodes.append(len(ci))
                ind = np.argsort(n_component_nodes)
                # print(ind)
                for i in ind[:-1]:
                    ci = component_indices[i]  # index of component_i
                    cj = np.concatenate(component_indices[:i] + component_indices[i + 1:])
                    if np.min(distance_matrix[ci, :][:, cj]) > 30:
                        labels_new[ci] = 0
        labels = labels_new

        ## 5. reomve the patches which the distance from its nearest component is larger than 30mm
        dilate_dist = 20

        template.remove_gifti_data_array(0)
        template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.array(labels).astype(np.int32)))
        nib.loadsave.save(template, post_label)

        p = subprocess.Popen("wb_command -label-dilate {} {} {} {}".format(
            post_label, surf_file, dilate_dist, post_label),
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return_code = p.wait()
        if return_code != 0:
            print('wb_command -label-dilate failed')
            exit(1)

        p = subprocess.Popen("wb_command -label-mask {} {} {}".format(
            post_label, group_label, post_label),
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return_code = p.wait()
        if return_code != 0:
            print('wb_command -label-mask failed')
            exit(1)


def post_proc_s12(sub_list_file, label_dir, method, hemi):
    sub_list = np.loadtxt(sub_list_file, dtype=str)
    print(sub_list.shape)

    # for hemi in ['L', 'R']:
    # for hemi in ['R']:
    # for hemi in ['L']:
    areas = surface.load_surf_data('fsaverage.{}.midthickness_areas.32k_fs_LR.func.gii'.format(hemi))
    adjacency_matrix = load_npz("adjacency_matrix_{}.npz".format(hemi))

    template = nib.load('fsaverage.{}.Glasser.32k_fs_LR.label.gii'.format(hemi))

    distance_matrix = np.load('dist_matrix_{}.npy'.format(hemi))
    predecessors = np.load('predecessors_{}.npy'.format(hemi))
    print(distance_matrix.shape, predecessors.shape)

    if 'Glasser' in method:
        group_label = 'fsaverage.{}.Glasser.32k_fs_LR.label.gii'.format(hemi)
    elif 'BN_Atlas' in method:
        group_label = 'fsaverage.{}.Glasser.32k_fs_LR.label.gii'.format(hemi)
    else:
        print("ERROR: altas not valid, only Glasser or BN_Atlas")
        return
    surf_file = 'fsaverage.{}.midthickness.32k_fs_LR.surf.gii'.format(hemi)

    for ni, sub in enumerate(sub_list):
        label_name = '{}/{}/{}_{}.32k_fs_LR.label.gii'.format(
            label_dir, sub, method, hemi)
        post_label = '{}/{}/{}_post12_{}.32k_fs_LR.label.gii'.format(label_dir, sub, method, hemi)
        if not os.path.exists(label_name):
            continue
        if os.path.exists(post_label):
            continue

        labels = surface.load_surf_data(label_name)
        print(ni, sub)

        ## 1. remove small patches
        labels_new = np.array(labels)
        for label in np.unique(labels)[1:]:
            label_indices = np.where(labels == label)[0]
            subgraph = adjacency_matrix[label_indices][:, label_indices]
            n_components, component_labels = connected_components(subgraph)

            for component in range(n_components):
                component_indices = np.where(component_labels == component)[0]
                component_nodes = label_indices[component_indices]
                if areas[component_nodes].sum() < 25:
                    labels_new[component_labels] = 0
        labels = labels_new

        ## 2. join nearby patch
        labels_new = np.array(labels)
        for label in np.unique(labels)[1:]:
            label_indices = np.where(labels == label)[0]
            subgraph = adjacency_matrix[label_indices][:, label_indices]
            n_components, component_labels = connected_components(subgraph)

            for i in range(n_components):
                ci = label_indices[np.where(component_labels == i)[0]]  # index of component_i
                for j in range(i + 1, n_components):
                    cj = label_indices[np.where(component_labels == j)[0]]  # index of component_j
                    cis = np.where(
                        distance_matrix[ci, :][:, cj] < 4)  # source (com_i) and target (com_j) nodes
                    # print(cis)
                    if cis[0].size == 0:
                        break
                    ii, jj = cis
                    src_ind = ci[ii]  # source (com_i) nodes
                    tgt_ind = cj[jj]  # target (com_i) nodes
                    for src, tgt in zip(src_ind, tgt_ind):
                        curr_node = tgt
                        while curr_node != src:
                            labels_new[curr_node] = label
                            curr_node = predecessors[src, curr_node]
        labels = labels_new

        ## 5. reomve the patches which the distance from its nearest component is larger than 30mm
        dilate_dist = 20

        template.remove_gifti_data_array(0)
        template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.array(labels).astype(np.int32)))
        nib.loadsave.save(template, post_label)

        p = subprocess.Popen("wb_command -label-dilate {} {} {} {}".format(
            post_label, surf_file, dilate_dist, post_label),
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return_code = p.wait()
        if return_code != 0:
            print('wb_command -label-dilate failed')
            exit(1)

        p = subprocess.Popen("wb_command -label-mask {} {} {}".format(
            post_label, group_label, post_label),
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return_code = p.wait()
        if return_code != 0:
            print('wb_command -label-mask failed')
            exit(1)


from sys import argv
if __name__ == '__main__':
    # sub_list_file, label_dir, method, hemi
    post_proc(argv[1], argv[2], argv[3], argv[4])
    # post_proc_s12(argv[1], argv[2], argv[3], argv[4])

