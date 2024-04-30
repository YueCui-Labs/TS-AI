import nibabel as nib
import numpy as np

import scipy.io as scio
from scipy.sparse import csr_matrix

import vtk
from sphericalunet.utils.vtk import read_vtk

import os, subprocess


def write_vtk(coord, faces, file):
    with open(file, 'w') as f:
        f.write("# vtk DataFile Version 3.0 \n")
        f.write("vtk output \n")
        f.write("ASCII \n")
        f.write("DATASET POLYDATA \n")
        f.write("POINTS {} float \n".format(str(len(coord))))
        np.savetxt(f, coord, fmt='%.4f')
        f.write("POLYGONS {} {} \n".format(str(len(faces)), str(faces.size)))
        np.savetxt(f, faces, fmt='%d')


adj10242 = scio.loadmat('adj_mat.mat')['adj_mat']
adj2562 = scio.loadmat('adj_mat_2562.mat')['adj_mat_2562']
adj642 = scio.loadmat('adj_mat_642.mat')['adj_mat_642']

sphere_40k = read_vtk('Sphere.40k.vtk')
vert_coord = sphere_40k['vertices']

for adj, vnum in zip([adj10242, adj2562, adj642], [10242, 2562, 642]):
    # adj = csr_matrix(adj)

    sphere_warped_vtk = os.path.join('Sphere.{}.vtk'.format(vnum))
    sphere_warped_gii = os.path.join('Sphere.{}.surf.gii'.format(vnum))

    # write sphere file and map feature/label 32k file to 40k file
    coords = vert_coord[:vnum, :]  # nonlinear warping
    print('adj sum: ', adj.sum())

    faces = []
    edge_stack = []
    i0 = 0
    j0 = np.where(adj[i0, :] == 1)[0][0]
    edge_stack.append([i0,j0])

    while edge_stack:
        i, j = edge_stack.pop()
        # if adj[i,j] == 0:
        #     continue

        ks = np.intersect1d(np.where(adj[j, :] == 1)[0], np.where(adj[:, i] == 1)[0], assume_unique=True)
        if len(ks) > 0:
            k = ks[0]
            # if adj[i, j]==1 and adj[j, k]==1 and adj[k,i]==1:
            # if not (adj[i, j] == 1 and adj[j, k] == 1 and adj[k, i] == 1):
                # print(i,j,k, adj[i, j], adj[j, k], adj[k, i])
                # continue
            faces.append([i, j, k])
            edge_stack.append([j, i])
            edge_stack.append([k, j])
            edge_stack.append([i, k])
            adj[i, j] = 0
            adj[j, k] = 0
            adj[k, i] = 0


    # for i in range(vnum):
    #     ddj = np.where(adj[i, :] == 1)[0]
    #     for j in ddj:
    #         ddk = np.intersect1d(np.where(adj[j,:] == 1)[0], np.where(adj[:,i] == 1)[0], assume_unique=True)
    #         for k in ddk:
    #             faces.append([i,j,k])
    #             adj[i,j] = 0
    #             adj[j,k] = 0
    #             adj[k,i] = 0
    faces = np.array(faces, dtype=np.int32)
    print(len(faces))  # 15787 x, 20480 v
    a, ind = np.unique(np.sort(faces), axis=0, return_index=True)
    print(ind.shape)
    faces = faces[ind]
    print(faces.shape)

    # Create a nibabel gifti image object
    surf_img = nib.gifti.GiftiImage()

    # Create a nibabel gifti data array object for the coordinates
    coords_data_array = nib.gifti.GiftiDataArray(coords, intent='NIFTI_INTENT_POINTSET')

    # Create a nibabel gifti data array object for the faces
    faces_data_array = nib.gifti.GiftiDataArray(faces, intent='NIFTI_INTENT_TRIANGLE')

    # Add the data arrays to the gifti image object
    surf_img.add_gifti_data_array(coords_data_array)
    surf_img.add_gifti_data_array(faces_data_array)

    # Save the gifti image object to a file
    nib.save(surf_img, sphere_warped_gii)

    # write_vtk(coords, faces, sphere_warped_vtk)
    # points = vtk.vtkPoints()
    # for coord in coords:
    #     points.InsertNextPoint(coord[0], coord[1], coord[2])
    #
    # triangles = vtk.vtkCellArray()
    # for face in faces:
    #     triangle = vtk.vtkTriangle()
    #     triangle.GetPointIds().SetId(0, face[0])
    #     triangle.GetPointIds().SetId(1, face[1])
    #     triangle.GetPointIds().SetId(2, face[2])
    #     triangles.InsertNextCell(triangle)
    #
    # polydata = vtk.vtkPolyData()
    # polydata.SetPoints(points)
    # polydata.SetPolys(triangles)
    #
    # writer = vtk.vtkPolyDataWriter()
    # writer.SetFileName(sphere_warped_vtk)
    # writer.SetInputData(polydata)
    # writer.Write()

    # p = subprocess.Popen('surf2surf -i {} -o {}'.format(sphere_warped_vtk, sphere_warped_gii),
    #                      shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # return_code = p.wait()
    # if return_code != 0:
    #     print("surf2surf failed!")
    #     exit(1)
