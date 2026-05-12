clear;
clc;

root_path = fileparts(mfilename('fullpath'));

% Put *.mat datasets in ./data by default. Each dataset should contain
% BPs, Y, and g.
data_path = fullfile(root_path, "data");
addpath(data_path);
lib_path = fullfile(root_path, "lib");
addpath(lib_path);
perf_path = fullfile(root_path, "FKCC");
addpath(perf_path);

fkcc_shared_mex = @FKCC_shared_gini1_mex;

exp_n = 'FKCC_shared_center_gini';


dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};


for i1 =1 : length(datasetCandi)
    data_name = datasetCandi{i1}(1:end-4);
    dir_name = [pwd, filesep, exp_n, filesep, data_name];
    try
        if ~exist(dir_name, 'dir')
            mkdir(dir_name);
        end
        prefix_mdcs = dir_name;
    catch
        disp(['create dir: ',dir_name, 'failed, check the authorization']);
    end
    
    
    clear X y Y g BPs;
    load(data_name, 'BPs', 'Y', 'g');
    y = Y;
    nCluster = length(unique(y));
    nSmp = length(y);
    nBase = 20;
    fname2 = fullfile(prefix_mdcs, [data_name,'_' exp_n, '.mat']);
    if ~exist(fname2, 'file')
        % **************************************************************************
        % Parameter Configuration
        % **************************************************************************
        
        nRepeat = 10;
        nMeasure = 27;
        maxIterFKCC = 100;
        
        seed = 2024;
        rng(seed, 'twister')
        
        % Generate 50 random seeds
        random_seeds = randi([0, 1000000], 1, nRepeat * nRepeat);
        
        % Store the original state of the random number generator
        original_rng_state = rng;
        
        FKCC_shared_center_gini_result = zeros(nRepeat, nMeasure);
        FKCC_shared_center_gini_iter = zeros(nRepeat, 1);
        
        
        t1_s = tic;
        t1 = toc(t1_s);
        t2_s = tic;
        for iRepeat = 1:nRepeat
            idx = (iRepeat - 1) * nBase + 1 : iRepeat * nBase;
            BPi = BPs(:, idx);
            
            n_inner_repeat = 1;
            for inner_repeat = 1:n_inner_repeat
                % Restore the original state of the random number generator
                rng(original_rng_state);
                % Set the seed for the current iteration
                rng(random_seeds( (iRepeat-1) * nRepeat + inner_repeat ));
                t = 20;
                [Hc_new, ~, ~] = compute_ECA_from_BPs_fast(BPi, t);

                [label_0, ~] = kmeanspp_v3_fast2(Hc_new', nCluster);
                [label_cell, iter_num] = fkcc_shared_mex(Hc_new, {label_0'}, g', maxIterFKCC, 0, nCluster);
                label = label_cell{1};
                FKCC_shared_center_gini_iter(iRepeat) = iter_num;
            end
            res_27 = my_eval_y_fair_mismatch_HAB(y, g, label);
            
            FKCC_shared_center_gini_result(iRepeat, :) = res_27;
            clear Hc_new label_0 label;
        end
        t2 = toc(t2_s);
        ts = [t1, t2];
        FKCC_shared_center_gini_result_time = t1 + t2/nRepeat;
        FKCC_shared_center_gini_result_mean = mean(FKCC_shared_center_gini_result, 1);
        FKCC_shared_center_gini_result_std =  std(FKCC_shared_center_gini_result, 1);
        save(fname2, 'FKCC_shared_center_gini_result_mean', 'FKCC_shared_center_gini_result', 'FKCC_shared_center_gini_result_time', 'FKCC_shared_center_gini_result_std', 'FKCC_shared_center_gini_iter', 'maxIterFKCC', 'ts');
        disp([data_name, ' has been completed!']);
    end
end
rmpath(data_path);
rmpath(lib_path);
rmpath(perf_path);
