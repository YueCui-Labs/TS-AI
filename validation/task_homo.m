function [task_inhomo_avg] = task_homo(sub_list_file, indiv_dir, tfmri_dir, method, cifti_vert_LR_file, output_dir)
% task inhomogeneity 
% task_inhomo('list_tfmri.txt', '../HCP1200', 'data_tfMRI', 'ic_anat', 'result')
addpath('/data0/user/cyli/matlab/GIFTI');
addpath('/data0/user/cyli/matlab/cifti-matlab');

task_num=7;
task_list={'EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM'};
contrast_ind_list={1:2, 1:2, 1:2, 1:6, 1:2, 1:2, 1:8};

%% read sub list
fid = fopen(sub_list_file);
sub_list = textscan(fid, '%s');
sub_list =sub_list{1};
fclose(fid);
sub_num = length(sub_list);

%% read group atlas
if contains(method, 'BN_Atlas')
    atlas = 'BN_Atlas';
elseif contains(method, 'Glasser')
    atlas = 'Glasser';
end
group_atlas_L_file=['../TS_AI/atlases/fsaverage.L.', atlas, '.32k_fs_LR.label.gii'];
group_atlas_R_file=['../TS_AI/atlases/fsaverage.R.', atlas, '.32k_fs_LR.label.gii'];

group_atlas_L_st = gifti(group_atlas_L_file);
group_atlas_L = group_atlas_L_st.cdata;

group_atlas_R_st = gifti(group_atlas_R_file);
group_atlas_R = group_atlas_R_st.cdata;

%% read start_L, count_L, vertlist_L, start_R, count_R, vertlist_R
load(cifti_vert_LR_file);

task_homo_mat=nan(sub_num,task_num, 8); % #sub x #task
for s=1:sub_num
    sub=sub_list{s};
    disp([num2str(s), ' ', sub])

    if contains(method, 'group')
        indiv_L=group_atlas_L;
        indiv_R=group_atlas_R;
        indiv_atlas = [group_atlas_L(:)',group_atlas_R(:)'];
    else
        label_L_file=fullfile(indiv_dir,sub, [method,'_L.32k_fs_LR.label.gii']);
        label_R_file=fullfile(indiv_dir,sub, [method,'_R.32k_fs_LR.label.gii']);
        if ~exist(label_L_file) || ~exist(label_R_file); continue; end

        %% read indiv atlas
        indiv_L_st = gifti(label_L_file);
        indiv_L=indiv_L_st.cdata;

        indiv_R_st = gifti(label_R_file);
        indiv_R=indiv_R_st.cdata;

        indiv_atlas = [indiv_L(:)',indiv_R(:)'];
    end
    vertex_num = length(indiv_atlas);
    uni_labels = unique(indiv_atlas(indiv_atlas>0));
    parcel_num = length(uni_labels);

    parcel_mask = zeros(parcel_num,vertex_num,'logical');
    parcel_vert_num = zeros(parcel_num,1);

    for l=1:parcel_num
        parcel_mask(l,:) = indiv_atlas==uni_labels(l);
        parcel_vert_num(l) = sum(parcel_mask(l,:));
    end

    task_homo_indiv=zeros(task_num);
    for task_ind=1:task_num
        task=task_list{task_ind};
        contrast_ind=contrast_ind_list{task_ind};
        contrast_num=length(contrast_ind);

        tfmri_file = fullfile(tfmri_dir, sub, [sub,'_tfMRI_',task,'_level2_hp200_s2_MSMAll.dscalar.nii']);  % TODO
        if ~exist(tfmri_file); continue; end

        %% read task activation
        task_activation_st=cifti_read(tfmri_file);
        task_activation=task_activation_st.cdata(:,contrast_ind); % 91282 x #contrast

        % left task activation
        contrast_all_vert_L = zeros(length(indiv_L), contrast_num);
        contrast_all_vert_L(1+vertlist_L,:) = task_activation(start_L:start_L+count_L-1,:);

        % right task activation
        contrast_all_vert_R = zeros(length(indiv_R), contrast_num);
        contrast_all_vert_R(1+vertlist_R,:) = task_activation(start_R:start_R+count_R-1,:);
        
        contrast = [contrast_all_vert_L; contrast_all_vert_R]; % #vertex x #contrast

        task_homo=nan(parcel_num, contrast_num);
        vert_valid_sum = 0;
        for l=1:parcel_num
            parcel_mask_l = parcel_mask(l,:);  % (1,vert)

            if parcel_vert_num(l) < 10; continue; end
            tmp_std = std(contrast(parcel_mask_l,:),1);  % (1,contrast)
            if min(tmp_std) < 1e-5; continue; end
            task_homo(l,:) = 1 ./ tmp_std .* parcel_vert_num(l); % 1 x #contrast
            vert_valid_sum = vert_valid_sum + parcel_vert_num(l);
        end
        % mean in weights
        if vert_valid_sum == 0 ; continue; end
        task_homo_mat(s,task_ind,contrast_ind) = sum(task_homo, 1,'omitnan') / vert_valid_sum;
    end
end

save(fullfile(output_dir,['04_task_homo_', method,'.mat']), 'task_homo_mat');

task_mean = squeeze(mean((mean(task_homo_mat,1, 'omitnan')), 2, 'omitnan'));
disp([method,'_task_contrast_homo'])
disp(task_mean)
disp(mean(task_mean))

