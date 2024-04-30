import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # file
    parser.add_argument('--reference_atlas', default='Glasser', type=str)
    parser.add_argument('--hemi', default='L', choices=['L', 'R'], type=str)
    parser.add_argument('--feat_dir', default='../features/HCP1200', type=str)
    parser.add_argument('--fold', type=str)
    parser.add_argument('--sub_num', type=int, default=388)

    parser.add_argument('--indiv_prename', type=str)
    parser.add_argument('--syn_prename', type=str)
    parser.add_argument('--syn_task_name', type=str)
    parser.add_argument('--loss_name', type=str, default='ce')
    
    # model parameters
    parser.add_argument('--modals', nargs='+', default=['d', 'r'], type=str)
    parser.add_argument('--out_modal', default='t', type=str)
    parser.add_argument('--out_channels', type=int, default=180)
    parser.add_argument('--save_dir', type=str, default='trained_models')
    parser.add_argument('--lamb', type=int, default=10)
    parser.add_argument('--lamb_syn', type=int, default=10)

    # training
    parser.add_argument('--train_sub_list_file', type=str)
    parser.add_argument('--vali_sub_list_file', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epoch_num', type=int, default=50)

    # testing
    parser.add_argument('--test_sub_list_file', type=str)
    parser.add_argument('--indiv_label_dir', default='../indiv_atlas/HCP1200', type=str)
    parser.add_argument('--sphere_32k', default='neigh_indices/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii', type=str)
    parser.add_argument('--sphere_40k', default='neigh_indices/Sphere.40k.surf.gii', type=str)

    # hardware setting
    parser.add_argument('--cuda', default=1, choices=[0, 1], type=int, help='Use GPU calculating')
    parser.add_argument('--num_workers', default=16, type=int, help='number of CPU core to be used of DataLoader')


    args = parser.parse_args()

    return args
