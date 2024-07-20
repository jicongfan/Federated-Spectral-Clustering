%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
%    5.2 Clustering Performance of FedSC on Iris and COIL20 Datasets    %
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

% Step 2: noise
% noise.gammas = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 1];
noise.gammas = [0, 0.1, 0.3, 0.5, 0.7, 0.25];
noise.xid = 6;
noise.zid = 1;
noise.cid = 1;
noise.flag_zc = 0;% perturbed Z and C by Gaussian noise with sigma_z and sigma_c; 1 for perturbing Z and C; 0 for no action
prt = 1;

%ds_indices = [1, 2];
ds_indices = 1;
ACCbox = zeros(length(ds_indices), 2);% row: datasets; col: methods
NMIbox = zeros(length(ds_indices), 2);% row: datasets; col: methods
tic
%% Experiment Setup
for gid = 1: length(ds_indices)
    %% PART I: Sensitivity analysis on concentric circles
    ds.id = ds_indices(gid);
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
    options.lambda = 1e-4;
    % options.sigma2 = classreg.learning.svmutils.optimalKernelScale(X,[],1);
    options.replicates = 5;% number of restarts for kmeans
    options.is_perturbed_by_factors = 0;
    options.expe_prt = 0;
    options.n_trials = 20;
    
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
    options.d_tol = 0.1;
    if noise.gammas(noise.xid) > 0 || noise.gammas(noise.zid) > 0 || noise.gammas(noise.cid) > 0
        [X, ~] = fedsc_dataloader(ds.name(ds.id));
        sigma2 = sigma2_estimator(X, options.rbf_c);
        options.d = d_estimator(X, sigma2, options.d_tol);
        clear xs sigma2;
    end

    % Step 5: run spectral clustering
    fprintf('\n');
    dispIteration('****** Step 2: Perform FedSC ******', prt);
    dispIteration('**************************************', prt);
    [ACCbox(gid, :), NMIbox(gid, :), Xn, Ls] = doSensitivityAnalysis(ds, noise, options);
    dispIteration('**************************************', prt);
    dispIteration('* Methods complete.', prt);
    
    fprintf('\n');


    % -----------------------------------
    Legends = ["GT", "SC", "FedSC"];
    show_modes = ["single", "stack"];
    if strcmp(ds.name(ds.id), 'COIL20')
        tmp_ds = load('COIL20_tsne.mat');
        tsneX = tmp_ds.tsneX;
        % plot_ClusterLr(tsneX, Ls, Legends, show_modes(2));
        figure;gscatter(tsneX(:, 1), tsneX(:, 2), Ls(:, 1));legend off;
        figure;gscatter(tsneX(:, 1), tsneX(:, 2), Ls(:, 2));legend off;
        figure;gscatter(tsneX(:, 1), tsneX(:, 2), Ls(:, 3));legend off;
    elseif strcmp(ds.name(ds.id), 'Iris')
        tmp_ds = load('Iris_tsne.mat');
        tsneX = tmp_ds.tsneX;
        % plot_ClusterLr(tsneX, Ls, Legends, show_modes(2));
        figure;gscatter(tsneX(:, 1), tsneX(:, 2), Ls(:, 1));legend off;
        figure;gscatter(tsneX(:, 1), tsneX(:, 2), Ls(:, 2));legend off;
        figure;gscatter(tsneX(:, 1), tsneX(:, 2), Ls(:, 3));legend off;
    else
        plot_ClusterLr(Xn', Ls, Legends, show_modes(2));
    end
end
toc