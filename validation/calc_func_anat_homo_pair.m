function calc_func_anat_homo_pair(method, rfMRI_dir, dti_dir, indiv_dir, sub_list_file, saveflg)

addpath('/data0/user/cyli/matlab/cifti-matlab');
addpath('/data0/user/cyli/matlab/GIFTI');


%% load reference (group) atlas
if contains(method, 'BN_Atlas')
    atlas = 'BN_Atlas';
elseif contains(method, 'Glasser')
    atlas = 'Glasser';
end
ref_L_st = gifti(['../TS_AI/atlases/fsaverage.L.', atlas, '.32k_fs_LR.label.gii']);
ref_R_st = gifti(['../TS_AI/atlases/fsaverage.R.', atlas, '.32k_fs_LR.label.gii']);
ref_L=ref_L_st.cdata;
ref_R=ref_R_st.cdata;
atlas_mask_L = ref_L>0;
atlas_mask_R = ref_R>0;

ref = [ref_L(atlas_mask_L); ref_R(atlas_mask_R)];

%% write session
session_list={'rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL'};
%rfMRI_dir='~/HCP/HCP1200/HCP_retest_cpfmri'; % out-of-sample
%dti_dir='~/HCP/HCP_retest';  % out-of-sample TODO
%indiv_dir='../indiv_atlas/HCP1200';

%% read start_L, count_L, vertlist_L, start_R, count_R, vertlist_R
cifti_vert_LR_file='cifti_vert_LR.mat';
load(cifti_vert_LR_file);
all_vert_num=32492;

%% read sub_list
sub_list=textscan(fopen(sub_list_file),'%s');
sub_list = sub_list{1};
sub_num=length(sub_list);


anat_homo = [];
func_homo = [];

for s=1:sub_num
    sub=sub_list{s};
    disp(sub);

    save_homo_file = fullfile(indiv_dir, sub, ['03_pair_dr_homo_', method, '.txt']);

    %% next if calc. have saved
    if exist(save_homo_file)
        tmp_homo = textscan(fopen(save_homo_file), '%f');
        tmp_homo = tmp_homo{1}; % cell --> array

        if length(tmp_homo) == 2
            anat_homo(end+1) = tmp_homo(1);
            func_homo(end+1) = tmp_homo(2);
            continue
        end
    end

    %% next if method generated atlas do not exist
    if ~contains(method, 'group')
        label_file_L = fullfile(indiv_dir, sub, [method ,'_L.32k_fs_LR.label.gii']);
        label_file_R = fullfile(indiv_dir, sub, [method ,'_R.32k_fs_LR.label.gii']);

        if ~exist(label_file_L) || ~exist(label_file_R); disp('label not found'); continue; end
    end

    %% read functional connectivity
    func_conn=[];
    for ff=1:length(session_list)

        session=session_list{ff};
        func_ts_file=fullfile(rfMRI_dir, sub, [session, '_Atlas_MSMAll_hp2000_clean.dtseries.nii'])

        if ~exist(func_ts_file); disp('rfmri not found'); continue; end

        function_ts_st=cifti_read(func_ts_file);
        cdata=function_ts_st.cdata; % 91282 x ts

        all_vert_ts_L = zeros(all_vert_num, size(cdata,2));
        all_vert_ts_L(1+vertlist_L,:) = cdata(start_L:start_L+count_L-1,:);

        all_vert_ts_R = zeros(all_vert_num, size(cdata,2));
        all_vert_ts_R(1+vertlist_R,:) = cdata(start_R:start_R+count_R-1,:);

        vali_vert_ts_LR = [all_vert_ts_L(atlas_mask_L, :); all_vert_ts_R(atlas_mask_R, :)];
        func_conn = [func_conn, vali_vert_ts_LR];
    end

    if length(func_conn) == 0; continue; end % next if func_conn is none
    size(func_conn);

    %% read anatomical connenctivity
    anat_file_L = fullfile(dti_dir, sub, 'ptx_NxN_32k_L_MSMAll', 'fdt_matrix3.mat');
    anat_file_R = fullfile(dti_dir, sub, 'ptx_NxN_32k_R_MSMAll', 'fdt_matrix3.mat');

    anat_conn_L = load(anat_file_L);
    anat_conn_L = anat_conn_L.conn_sp;

    anat_conn_R = load(anat_file_R);
    anat_conn_R = anat_conn_R.conn_sp;
    anat_conn = [anat_conn_L; anat_conn_R];

    anat_conn = log2(full(anat_conn)+1);
    anat_conn = anat_conn([atlas_mask_L; atlas_mask_R], :); % 29696+29716 x #MESH

    size(anat_conn);

    % calc. homo.
    if contains(method, 'group')
        indiv_lab = ref;
    else

        label_L_st = gifti(label_file_L);
        label_L = label_L_st.cdata;
%        unique(label_L)

        label_R_st = gifti(label_file_R);
        label_R = label_R_st.cdata;
%        unique(label_R)

        indiv_lab = [label_L(atlas_mask_L); label_R(atlas_mask_R)];
    end

    % compute indiv homo
    anat_homo(end+1) = compute_homo(anat_conn, indiv_lab);
    func_homo(end+1) = compute_homo(func_conn, indiv_lab);

    %% save temp. homo.
    fprintf(fopen(save_homo_file,'w'), '%f %f', anat_homo(end), func_homo(end));

end

disp([method, '_rs_dti_homo'])
mfunc_homo = mean(func_homo)
manat_homo = mean(anat_homo)

if saveflg == 1
   save(['result/03_pair_dr_homo_', method, '.mat'],'func_homo', 'anat_homo');
end

end


function S_w = compute_homo(conn_mat, atlas)
% compute SS_between / SS_within
% SS_within = mean of distance between vertices within the parcel and the centriod
% large --> parcels has been improved
%    size(conn_mat)
%    size(atlas)
    parc_num = max(atlas);

    mu = nan(parc_num,size(conn_mat, 2));
    S_w = nan(parc_num,1);
    areal = sum(atlas>0);

    for parc=1:parc_num
        if sum(atlas==parc)==0; continue; end
%        mu(parc,:) = mean(conn_mat(atlas==parc,:), 1,'omitnan'); %(1,d)
%        S_w(parc) = mean(1-pdist2(conn_mat(atlas==parc,:), mu(parc,:), 'correlation'), 'omitnan');
        S_w(parc) = mean(1 - pdist(conn_mat(atlas==parc,:), 'correlation'), 'omitnan');
        S_w(parc) = S_w(parc) * sum(atlas==parc)/ areal;
    end
    S_w = sum(S_w,'omitnan');
end