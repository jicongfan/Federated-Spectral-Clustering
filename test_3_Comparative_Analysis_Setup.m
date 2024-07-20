%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
% 5.3 Comparative analysis with existing distributed clustering methods %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% clear Workspace and Command Window
clc
clear all
warning off

% include the required working directories
addpath('datasets');
addpath('datasets/mnist/');
addpath('datasets/cifar10/cifar-10-batches-mat/');
addpath('models');
addpath('op_utils/');
addpath(genpath('utils'));
rmpath('utils/backup/');


%% Experiment Setup
% Step 1: dataset
ds.name = ["CC", "Iris", "COIL20", "Bank", "USPS", "ORL", "mnist", "cifar10"];
ds.id = 7;

% Step 2: noise
% noise.gammas = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 1];
noise.gammas = [0, 0.1, 0.3, 0.5, 0.7, 0.9];
noise.xid = 6;
noise.zid = 1;
noise.cid = 1;
noise.flag_zc = 0;% perturbed Z and C by Gaussian noise with sigma_z and sigma_c; 1 for perturbing Z and C; 0 for no action
prt = 1;

noise_indices = [1, 3, 5, 6];
% noise_indices = 1;
meanACC = zeros(length(noise_indices), 6);
stdACC = zeros(length(noise_indices), 6);
for gid = 1:length(noise_indices)
    noise.xid = noise_indices(gid);
    fprintf('******* Group %d: [ds_name x_sigma z_sigma c_sigma] = [%s %.2f %.2f %.2f]\n', gid, ds.name(ds.id), noise.gammas(noise.xid), noise.gammas(noise.zid), noise.gammas(noise.cid));
    dispIteration('*********************************************', prt);
    % Step 3: print info of dataset and noisy perturbation
    dispIteration('****** Step 1: Show the configurations ******', prt);
    % dispIteration(['****** Loaded dataset: ', ds.name(ds.id), '_tsne.mat.'], prt);
    fprintf('****** Loaded dataset: %s_tsne.mat.\n', ds.name(ds.id));
    dispIteration(['****** X_Noise_gamma of X_n = ', num2str(noise.gammas(noise.xid))], prt);
    dispIteration(['****** Z_Noise_gamma of Z_n = ', num2str(noise.gammas(noise.zid))], prt);
    dispIteration(['****** C_Noise_gamma of C_n = ', num2str(noise.gammas(noise.cid))], prt);
    
    % Step 4: Hyperparameter settings
    options.S = 20;% maximum of rounds for the communication between the central server and clients
    options.rbf_c = 1;
    options.prt = 0;
    options.clustering = 1;
    options.nclients = 8;
    options.lambda = 1e-3;
    % options.sigma2 = classreg.learning.svmutils.optimalKernelScale(X,[],1);
    options.replicates = 5;% number of restarts for kmeans
    options.is_perturbed_by_factors = 0;
    options.attacker = 1;
    options.expe_prt = 0;
    options.n_trials = 5;
    
    % dimension of the dictionary Z
    % d = 1;% for blob datasets
    % d = 1;% for concentric circles
    % d = 30;% for iris
    % d = 16^2;% for coil20
    % d = 10;% for Bank
    % d = 10;% for USPS
    % d = 10;% for Yeast
    % d = struct('Iris', 30, 'COIL20', 256, 'Mice', 500, 'USPS', 256);
    % options.d = d.(ds_name(ds_id));
    % options.d = 169;
    % dispIteration(['****** d = ', num2str(options.d)], 1);
    options.d_tol = 0.99;
    if noise.gammas(noise.xid) > 0 || noise.gammas(noise.zid) > 0 || noise.gammas(noise.cid) > 0
        [X, ~] = fedsc_dataloader(ds.name(ds.id));
        sigma2 = sigma2_estimator(X, options.rbf_c);
        options.d = d_estimator(X, sigma2, options.d_tol);
        clear xs sigma2;
    end
    
    % Step 5: Run 5 repeated trials
    fprintf('\n');
    dispIteration('****** Step 2: print epoch info ******', prt);
    dispIteration('**************************************', prt);
    n_trials = options.n_trials;
    ACC = zeros(n_trials, 6);
    NMI = zeros(n_trials, 6);
    for i = 1: n_trials
        fprintf('****** Epoch %d: ', i);
        % dispIteration(['Epoch ', num2str(i), ': [ds_name ] = ', ds.name(ds.id)], prt);
        [ACC(i, :), NMI(i, :), L, Lr] = doComparativeAnalysis(ds, noise, options);
        % fprintf('Epoch %d: [ds_name is_noise ] = [%s]\n', i, ds.name(ds.id));
    end
    dispIteration('**************************************', prt);
    dispIteration('* Methods complete.', prt);
    
    fprintf('\n');
    
    dispIteration('****** Step 3: print epoch info ******', 1);
    
    % ACC
    meanACC(gid, :) = mean(ACC, 1);
    stdACC(gid, :) = std(ACC, 0, 1);
    
    % NMI
    % meanNMI = mean(NMI)
    % stdNMI = std(NMI)
    
    dispIteration('------------------------------------------------------------------------------------', prt);
    dispIteration('Epoch    K-Means     SC      DSC     FedSC(P = 8)    FedSC(P = 1)    FedSC(Attacker)', prt);
    dispIteration('------------------------------------------------------------------------------------', prt);
    for epoch = 1: n_trials
        fprintf('  %d      %.4f    %.4f  %.4f      %.4f          %.4f           %.4f\n', epoch, ACC(epoch, 1), ACC(epoch, 2), ACC(epoch, 3), ACC(epoch, 4), ACC(epoch, 5), ACC(epoch, 6));
    end
    dispIteration('------------------------------------------------------------------------------------', prt);
    dispIteration(['Group ', num2str(gid), ' Complete.'], prt);
    fprintf('\n');
end

% Step 6: Print acc info
dispIteration('****** Step 4: print epoch info ******', 1);
dispIteration('------------------------------------------------------------------------------------------------', prt);
dispIteration('   Group    Stats    K-Means     SC      DSC     FedSC(P = 8)    FedSC(P = 1)    FedSC(Attacker)', prt);
dispIteration('------------------------------------------------------------------------------------------------', prt);
for gid = 1:length(noise_indices)
    fprintf('    %.1f     Mean     %.4f    %.4f  %.4f      %.4f          %.4f           %.4f\n', noise.gammas(noise_indices(gid)), meanACC(gid, 1), meanACC(gid, 2), meanACC(gid, 3), meanACC(gid, 4), meanACC(gid, 5), meanACC(gid, 6));
    fprintf('             Std     %.4f    %.4f  %.4f      %.4f          %.4f           %.4f\n', stdACC(gid, 1), stdACC(gid, 2), stdACC(gid, 3), stdACC(gid, 4), stdACC(gid, 5), stdACC(gid, 6));
    dispIteration('------------------------------------------------------------------------------------------------', prt);
end
fprintf('* Info: [ds_name x_sigma z_sigma c_sigma lambda] = [%s %.2f %.2f %.2f %.4f]\n', ds.name(ds.id), noise.gammas(noise.xid), noise.gammas(noise.zid), noise.gammas(noise.cid), options.lambda);

