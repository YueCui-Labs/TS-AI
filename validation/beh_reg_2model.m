%function acc = vali_regression(sub_list_file, pre_dir, method, behavior_file, repeat, kfold, lambda, output_dir)
%KERNEL_RIDGE_REGRESSION 
% Behavior = Dice * W
% argmin_W  || Dice * W - Behavior ||^2 + lambda * || W ||^2

% Li, Chengyi, 2022.3.6

% -----------inputs-------------
% sub_list_file
% pre_dir: load individual atlas
% method: ic / mmi
% behavior_file: #sub x #behavior
% repeat: default 100
% kfold: default 10
% lambda: regularization, default 1
% output_dir: save dice matrix & behavoir accuracy

function beh_reg_2model(method, sub_list_file1, sub_list_file2)
pre_dir='../indiv_atlas/HCP1200';
behavior_file='6Cat_beh59_sub1180.txt';
beh_sub_file='6Cat_sub1180.txt';
repeat=100; kfold=5; lambda=1;
output_dir='result';

addpath('/data0/user/cyli/matlab/GIFTI');

beh_sub_list = textscan(fopen(beh_sub_file), '%s');
beh_sub_list = beh_sub_list{1};

behavior = load(behavior_file);
behavior = zscore(behavior);

%% validation: kernel ridge regression
acc =  [];
pval = [];

%% read sub list
        for sub_list_file={sub_list_file1, sub_list_file2}
    sub_list = textscan(fopen(sub_list_file{1}), '%s');
    sub_list =sub_list{1};

    [~,beh_sub_mask, label_sub_mask] = intersect(beh_sub_list, sub_list);
    sub_list = sub_list(label_sub_mask);
    sub_num = length(sub_list);

%    if exist(fullfile(output_dir,[method,'_dice.mat']))
%        load(fullfile(output_dir,[method,'_dice.mat']));  % 'dice_coef','sub_mask'
%        sub_num=size(dice_coef,1);
%    else
        %% read indiv atlas
        sub_vert=[]; % #sub x #vertices
        for s=1:sub_num
            sub=sub_list{s};

            label_L_file=fullfile(pre_dir,sub,[method,'_L.32k_fs_LR.label.gii']);
            label_R_file=fullfile(pre_dir,sub,[method,'_R.32k_fs_LR.label.gii']);

            if exist(label_L_file) && exist(label_R_file)

                indiv_L_st = gifti(label_L_file);
                indiv_L=indiv_L_st.cdata;

                indiv_R_st = gifti(label_R_file);
                indiv_R=indiv_R_st.cdata;

                sub_vert(end+1,:) = [indiv_L(:)',indiv_R(:)'];
            end
        end

        atlas_vertices=max(sub_vert,[],1)>0;
        sub_vert=sub_vert(:,atlas_vertices);

        %% compute Dice #sub x #sub
        dice_coef=zeros(sub_num,sub_num);
        for i=1:sub_num
            for j=i+1:sub_num
                dice_coef(i,j)=mean(sub_vert(i,:)==sub_vert(j,:));
            end
        end
        dice_coef=dice_coef+dice_coef'+eye(sub_num); % upper tri --> sym matrix

%        if sub_num>0
%            save(fullfile(output_dir,[method,'_dice.mat']), 'dice_coef','sub_mask');
%        end
%    end

    %% read behave scores
    sub_beh = behavior(beh_sub_mask,:);
    beh_num = size(sub_beh,2);

    %% validation: kernel ridge regression
    acc1m =  zeros(repeat, kfold, beh_num);
    pval1m = zeros(repeat, kfold, beh_num);

    for rp=1:repeat
        %ramdomly seperate the data into k folds
        indices = crossvalind('Kfold', sub_num, kfold);

    %    pred = zeros(sub_num, beh_num);
        for fold = 1 : kfold
            test = (indices == fold);
            train = ~test;
            % kernel ridge regression
            % train: W = (D + lambda * I)^-1 C
            %        NxB  NxN                NxB
            % test:  pred = D2 * W
            %        MxB    MxN  NxB
            % 1 for test, k-1 for training

            train_dice = dice_coef(train, train); % D
            train_beh = sub_beh(train, :); % C

            test_dice = dice_coef(test, train); % D2
            test_beh = sub_beh(test, :); % truth

            W = ( train_dice + lambda*eye(sum(train)) ) \ train_beh;
            pred = test_dice * W; % #test x #beh

    %        size(pred)
    %        size(test_beh)

            for measure=1:beh_num
                [acc1m(rp, fold, measure), pval1m(rp, fold, measure)] = corr(pred(:,measure),test_beh(:,measure));
            end
        end
    end
    acc = cat(1, acc, acc1m);
    pval = cat(1, pval, pval1m);
end

size(acc)

avg_acc = squeeze(mean(mean(acc,1,'omitnan'),2,'omitnan'));
avg_pv = zeros(beh_num,1);
for measure=1:beh_num
    avg_pv(measure) = CBIG_corrected_resampled_ttest_kfoldCV(squeeze(acc(:,:,measure)),0);
end

disp(mean(avg_acc))
disp(mean(avg_acc(avg_pv<0.05)))
disp(mean(avg_acc(avg_pv<0.01)))

save(fullfile(output_dir,sprintf('02_%s_6cat_acc_restt_pval.mat', method)), 'avg_acc', 'avg_pv', 'acc', 'pval');
