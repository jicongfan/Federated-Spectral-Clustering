function [acc, nmi, L, Lr] = doComparativeAnalysis(ds, noise, options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
% 5.3 Comparative analysis with existing distributed clustering methods %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Script Summary
% Task: evaluate whether the cluster task has been done successfully
% Dataset: concentric circles (synthetic) and COIL20 (real-world)
% 

%% clear Workspace and Command Window
% clc
% clear all
% warning off

%% Step 1: copy hyperparameters
expe_prt = options.expe_prt;% turn off print_info
P = options.nclients;
is_loadXmask = 0;

%% Step 2: load data
dispIteration('****** Data matrix is loading ...', expe_prt);
[X, Label] = fedsc_dataloader(ds.name(ds.id));
dispIteration('Loaded dataset: ' + ds.name(ds.id) + '_tsne.mat.', 1);
% extract some info
[~, n] = size(X);
L = Label;
K = length(unique(Label));% number of clusters
Lr = zeros(length(L), 4);% store cluster assignments of four methods

%% Step 3: inject noise
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

%% Step 4: partition data into P clients
tic
dispIteration('****** Partition all data points into P clients... ', expe_prt);
if is_loadXmask == 0
    % Xmask = splitData2Clients(n, P, K + 2);
    Xmask = splitData2Clients(n, P, 'equal');
else
    load(ds.name(ds.id) + '_Xmask.mat');
end
dispIteration('Done.', expe_prt);


%% Step 5: go
dispIteration('****** Comparative analysis starts ******', expe_prt);
% create variables err and nmi for restoring the metric info of clustering
% performance
acc = zeros(1, 6);% row: datasets; col: methods -> 1 for kmeans; 2 for sc; 3 for dsc; 4 for fedsc (p = 8); 5 for fedsc (p = 1); 6 for fedsc (attacker)
nmi = zeros(1, 6);% row: datasets; col: methods -> 1 for kmeans; 2 for sc; 3 for dsc; 4 for fedsc (p = 8); 5 for fedsc (p = 1); 6 for fedsc (attacker)
%% Method I: K-Means
if 1
    dispIteration('****** perform KMeans ... ', expe_prt);
    Lr(:, 1) = kmeans(Xn', K, 'Replicates', options.replicates);
    % Lr(:, 1) = bestMap(L,Lr(:, 1));
    % [acc(ds_id, 1), nmi(ds_id, 1)] = cluster_metrics(L, Lr(:, 1));
    [acc(1, 1), nmi(1, 1)] = cluster_metrics(L, Lr(:, 1));
    dispIteration('Done.', expe_prt);
end


%% Method III: Distributed Spectral Clustering (DSC)
if 1
    dispIteration('****** perform DSC ... ', expe_prt);
    Lr(:, 3) = DSC(Xn', Xmask, K, options);
    % Lr(:, 3) = bestMap(L,Lr(:, 3));
    % [acc(ds_id, 3), nmi(ds_id, 3)] = cluster_metrics(L, Lr(:, 3));
    [acc(1, 3), nmi(1, 3)] = cluster_metrics(L, Lr(:, 3));
    dispIteration('Done.', expe_prt);
end

%% Method IV: Proposed FedSC
dispIteration('****** perform FedSC ... ', expe_prt);
if noise.flag_zc
    % turn on the mode: perturb factors
    options.is_perturbed_by_factors = 1;

    dispIteration('****** Achieving DP-FedSC by perturbing Z and C ...', 1);
    options.z_noise_ratio = noise.gammas(noise.zid);
    dispIteration(['****** options.Z_Noise_gamma = ', num2str(options.z_noise_ratio)], expe_prt);
    
    options.c_noise_ratio = noise.gammas(noise.cid);
    dispIteration(['****** options.C_Noise_gamma = ', num2str(options.c_noise_ratio)], expe_prt);
end

% -----------------------------------
% for P = 8
[cellLr, ~, ~, options] = FedSC(Xn, Xmask, K, options);
Lr(:, 4) = cellLr{1, 2};
% Lr(:, 4) = bestMap(L,Lr(:, 4));
% [acc(ds_id, 4), nmi(ds_id, 4)] = cluster_metrics(L, Lr(:, 4));
[acc(1, 4), nmi(1, 4)] = cluster_metrics(L, Lr(:, 4));

if options.attacker
    Lr(:, 6) = cellLr{1, 3};% attacker
    [acc(1, 6), nmi(1, 6)] = cluster_metrics(L, Lr(:, 6));% attacker
    % options = rmfield(options, ["d", "sigma2"]);
end

% for P = 1
tmp_nclients = options.nclients;
options.nclients = 1;
Xmask = splitData2Clients(n, options.nclients, 'equal');
[cellLr, ~, ~, options] = FedSC(Xn, Xmask, K, options);
Lr(:, 5) = cellLr{1, 2};
% Lr(:, 4) = bestMap(L,Lr(:, 4));
% [acc(ds_id, 4), nmi(ds_id, 4)] = cluster_metrics(L, Lr(:, 4));
[acc(1, 5), nmi(1, 5)] = cluster_metrics(L, Lr(:, 5));
options.nclients = tmp_nclients;
clear tmp_nclients;
dispIteration('Done.', expe_prt);


%% Method II: Vanilla spectral clustering (SC)
if 1
    dispIteration('****** perform vanilla SC ... ', expe_prt);
    % construct kernel matrix
    tKxx = kernel(Xn, Xn, options.sigma2);% ground truth of similarity matrix with identical hyperparameter of kernel function
    
    % make it sparse based on KNN operation
    numKnn = max(ceil(log(n)), 1);
    knn_type = 'complete';
    knn_tKxx = getSimilarityMatrixFromKernelMatrix(tKxx, numKnn, knn_type);
    
    % perform spectral clustering
    Lr(:, 2) = SpectralClustering(knn_tKxx, K);
    % Lr(:, 2) = bestMap(L,Lr(:, 2));
    % [acc(ds_id, 2), nmi(ds_id, 2)] = cluster_metrics(L, Lr(:, 2));
    [acc(1, 2), nmi(1, 2)] = cluster_metrics(L, Lr(:, 2));
    dispIteration('Done.', expe_prt);
end

end