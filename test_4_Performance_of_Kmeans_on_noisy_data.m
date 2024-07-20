%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
% 5.4 Performance of Kmeans on noisy data with different intensity of variance %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
ds.id = 3;

% Step 2: noise
% noise.gammas = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 1];
noise.gammas = [0, 0.1, 0.3, 0.5, 0.7, 0.25];
% noise.xid = 6;
noise.zid = 1;
noise.cid = 1;
noise.flag_zc = 0;% perturbed Z and C by Gaussian noise with sigma_z and sigma_c; 1 for perturbing Z and C; 0 for no action
prt = 1;
expe_prt = 0;

ds_indices = 1:8;%[1, 2, 3, 8];
noise_indices = [1, 3, 5];
% noise_indices = 1;
% meanACC = zeros(length(noise_indices), 6);
% stdACC = zeros(length(noise_indices), 6);
ACC = zeros(length(noise_indices), length(ds_indices));
NMI = zeros(length(noise_indices), length(ds_indices));
for gid = 1:length(noise_indices)
    for dsid = 1:length(ds_indices)
        noise.xid = noise_indices(gid);
        ds.id = ds_indices(dsid);
        fprintf('******* Group %d: [ds_name x_sigma z_sigma c_sigma] = [%s %.2f %.2f %.2f]\n', gid, ds.name(ds.id), noise.gammas(noise.xid), noise.gammas(noise.zid), noise.gammas(noise.cid));
        dispIteration('*********************************************', prt);
        % Step 3: print info of dataset and noisy perturbation
        dispIteration('****** Step 1: Show the configurations ******', prt);
        % dispIteration(['****** Loaded dataset: ', ds.name(ds.id), '_tsne.mat.'], prt);
        fprintf('****** Loaded dataset: %s_tsne.mat.\n', ds.name(ds.id));
        dispIteration(['****** X_Noise_gamma of X_n = ', num2str(noise.gammas(noise.xid))], prt);
        
        % Step 5: Run 5 repeated trials
        fprintf('\n');
        dispIteration('****** Step 2: print epoch info ******', prt);
        dispIteration('**************************************', prt);
        % load dataset
        [X, Label] = fedsc_dataloader(ds.name(ds.id));
        dispIteration('Loaded dataset: ' + ds.name(ds.id) + '_tsne.mat.', 1);
        % extract some info
        [~, n] = size(X);
        L = Label;
        K = length(unique(Label));% number of clusters

        % Step 3: inject noise
        if noise.gammas(noise.xid) > 0
            dispIteration('****** Gaussian noise is being injected into data matrix ...', expe_prt);
            dispIteration(['****** Noise_gamma = ', num2str(noise.gammas(noise.xid))], expe_prt);
            Xn = addNoise2X(X, noise.gammas(noise.xid));
            dispIteration('****** Done.', expe_prt);
        else
            Xn = X;
            dispIteration('****** No noise for X.', expe_prt);
        end
        
        dispIteration('Done.', expe_prt);

        % Step 5: go
        dispIteration('****** Comparative analysis starts ******', expe_prt);
        Lr = kmeans(Xn', K, "Replicates", 5);
        [ACC(gid, dsid), NMI(gid, dsid)] = cluster_metrics(L, Lr);
        dispIteration('**************************************', prt);
        dispIteration('* Methods complete.', prt);
        
        fprintf('\n');
    end
end

% Step 6: Print acc info
dispIteration('****** Step 4: print result info ******', 1);
dispIteration('-----------------------------------------------------------------------------------------', prt);
dispIteration('   Group      CC       Iris     COIL20     Bank      USPS      ORL      MNIST     CIFAR10', prt);
dispIteration('-----------------------------------------------------------------------------------------', prt);
for gid = 1:length(noise_indices)
    fprintf('    %.1f     %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f\n', noise.gammas(noise_indices(gid)), ACC(gid, 1), ACC(gid, 2), ACC(gid, 3), ACC(gid, 4), ACC(gid, 5), ACC(gid, 6), ACC(gid, 7), ACC(gid, 8));
    dispIteration('-----------------------------------------------------------------------------------------', prt);
end
% fprintf('* Info: [ds_name x_sigma] = [%s %.2f]\n', ds.name(ds.id), noise.gammas(noise.xid));

