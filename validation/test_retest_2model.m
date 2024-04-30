function test_retest_2model(method, sub_list_file1, sub_list_file2)
%test_retest('mmi_l0.04_w0.03', '/n02dat01/users/cyli/HCP/HCP_retest/sub.txt')
addpath('/data0/user/cyli/matlab/GIFTI');

sub_list1 = textscan(fopen(sub_list_file1),'%s');
sub_list1 = sub_list1{1};
sub_num1 = length(sub_list1);

sub_list2 = textscan(fopen(sub_list_file2),'%s');
sub_list2 = sub_list2{1};
sub_num2 = length(sub_list2);


test_dir = '../indiv_atlas/HCP1200';
retest_dir = '../indiv_atlas/HCP_retest';

within = zeros(sub_num1 + sub_num2, 1);
label_test = [];
label_retest = [];
for s=1:sub_num1
    sub = sub_list1{s};

    label_test_L = gifti(fullfile(test_dir, sub, [method,'_L.32k_fs_LR.label.gii']));
    label_test_L = label_test_L.cdata;
    label_test_R = gifti(fullfile(test_dir, sub, [method,'_R.32k_fs_LR.label.gii']));
    label_test_R = label_test_R.cdata;
    label_test(s,:) = [label_test_L; label_test_R];

    label_retest_L = gifti(fullfile(retest_dir, sub, [method,'_L.32k_fs_LR.label.gii']));
    label_retest_L = label_retest_L.cdata;
    label_retest_R = gifti(fullfile(retest_dir, sub, [method,'_R.32k_fs_LR.label.gii']));
    label_retest_R = label_retest_R.cdata;
    label_retest(s,:) = [label_retest_L; label_retest_R];

    within(s) = dice(label_test(s,:), label_retest(s,:));
end

for s=1:sub_num2
    sub = sub_list2{s};

    label_test_L = gifti(fullfile(test_dir, sub, [method,'_L.32k_fs_LR.label.gii']));
    label_test_L = label_test_L.cdata;
    label_test_R = gifti(fullfile(test_dir, sub, [method,'_R.32k_fs_LR.label.gii']));
    label_test_R = label_test_R.cdata;
    label_test(sub_num1+s,:) = [label_test_L; label_test_R];

    label_retest_L = gifti(fullfile(retest_dir, sub, [method,'_L.32k_fs_LR.label.gii']));
    label_retest_L = label_retest_L.cdata;
    label_retest_R = gifti(fullfile(retest_dir, sub, [method,'_R.32k_fs_LR.label.gii']));
    label_retest_R = label_retest_R.cdata;
    label_retest(sub_num1+s,:) = [label_retest_L; label_retest_R];

    within(sub_num1+s) = dice(label_test(sub_num1+s,:), label_retest(sub_num1+s,:));
end

between = zeros(sub_num1*(sub_num1-1)*2 + sub_num2*(sub_num2-1)*2,1);
k=0;
for i=1:sub_num1
    sub = sub_list1{i};
    for j=i+1:sub_num1
        between(k+1) = dice(label_test(i,:), label_test(j,:));
        between(k+2) = dice(label_retest(i,:), label_retest(j,:));
        between(k+3) = dice(label_retest(i,:), label_test(j,:));
        between(k+4) = dice(label_test(i,:), label_retest(j,:));
        k=k+4;
    end
end

for i=1:sub_num2
    sub = sub_list2{i};
    for j=i+1:sub_num2
        between(k+1) = dice(label_test(sub_num1+i,:), label_test(sub_num1+j,:));
        between(k+2) = dice(label_retest(sub_num1+i,:), label_retest(sub_num1+j,:));
        between(k+3) = dice(label_retest(sub_num1+i,:), label_test(sub_num1+j,:));
        between(k+4) = dice(label_test(sub_num1+i,:), label_retest(sub_num1+j,:));
        k=k+4;
    end
end

fprintf(fopen(sprintf('result/01_%s_2model_intra.txt',method),'w'), '%f\n', within);
fprintf(fopen(sprintf('result/01_%s_2model_inter.txt',method),'w'), '%f\n', between);

disp([method, '_test_retest'])
mean(within)
mean(between)
diff = mean(within) - mean(between)
cohend = (mean(within) - mean(between)) / sqrt( (var(within)+var(between)) / 2)

end


function dice_avg = dice(label1, label2)
   uni = unique([label1(label1>0), label2(label2>0)]);
   uni_num = length(uni);
   dice = zeros(uni_num, 1);
   for l =1:uni_num
       dice(l) = 2* sum(label1 == uni(l) & label2 == uni(l)) ./ (sum(label1 == uni(l)) + sum(label2 == uni(l))+ eps);
   end
    dice_avg = mean(dice);
end
