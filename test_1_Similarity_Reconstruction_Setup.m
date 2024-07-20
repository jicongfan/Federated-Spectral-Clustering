%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
%    5.1 Similarity Reconstruction via Federated Space Factorization    %
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

%% Step 1: load dataset
% Step 1: dataset
ds.name = ["CC", "Iris", "COIL20", "Bank", "USPS", "ORL", "mnist", "cifar10"];
ds.id = 3;

% Step 2: noise
% noise.gammas = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 1];
noise.gammas = [0, 0.1, 0.3, 0.5, 0.7, 2.5];
% noise.xid = 1;
noise.zid = 1;
noise.cid = 1;
noise.flag_zc = 0;% perturbed Z and C by Gaussian noise with sigma_z and sigma_c; 1 for perturbing Z and C; 0 for no action
prt = 1;


noise_indices = 1;%[1, 3, 5];
% noise_indices = 1;
% meanACC = zeros(length(noise_indices), 6);
% stdACC = zeros(length(noise_indices), 6);
bulletinboard = zeros(length(noise_indices), 5);% col 1 for acc; col 2 for nmi; col 3 for err_Kxx; col 4 for err_Kxx-sparse; col 5 for errKxxbd
options.S = 20;% maximum of rounds for the communication between the central server and clients
obj = zeros(options.S, length(noise_indices));
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
    options.rbf_c = 1;
    options.prt = 0;
    options.clustering = 1;
    options.nclients = 8;
    options.lambda = 1e-4;
    % options.sigma2 = classreg.learning.svmutils.optimalKernelScale(X,[],1);
    options.replicates = 5;% number of restarts for kmeans
    options.is_perturbed_by_factors = 0;
    options.expe_prt = 0;
    options.n_trials = 10;
    
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
    
    options.d = 19;
    % Step 5: Perform federated similarity reconstruction
    fprintf('\n');
    dispIteration('****** Step 2: Perform FedSC ******', prt);
    dispIteration('**************************************', prt);
    [bulletinboard(gid, 1), bulletinboard(gid, 2), bulletinboard(gid, 3:5), obj(:, gid), cellKxx, L, Xs] = doSimilarityReconstruction(ds, noise, options);
    dispIteration('**************************************', prt);
    dispIteration('* Methods complete.', prt);
    
    fprintf('\n');

    %
end

dispIteration('****** Step 3: print epoch info ******', 1);
dispIteration('--------------------------------------------------------', prt);
dispIteration("ds_name    Noise     ACC      NMI     errKxx    errKxxbd", prt);
dispIteration('--------------------------------------------------------', prt);
for gid = 1: length(noise_indices)
    noise.xid = noise_indices(gid);
    fprintf(' %s       %.1f    %.4f    %.4f   %.4f    %.4f\n', ds.name(ds.id), noise.gammas(noise.xid), bulletinboard(gid, 1), bulletinboard(gid, 2), bulletinboard(gid, 3), bulletinboard(gid, 5));
end
dispIteration('--------------------------------------------------------', prt);
dispIteration("* errKxx: nrm(hat_Kxx - tKxx, 'fro')/nrm(tKxx, 'fro')", prt);
dispIteration("* errKxxbd: \|\hat{K}_{\tilde{x},\tilde{x}} - K_{x,x}\|_\infty", prt);

% figure;gscatter(Xs{1,2}(1, :)', Xs{1,2}(2, :)', L, 'rc');legend({'1','2'}, 'Location', 'northeast');