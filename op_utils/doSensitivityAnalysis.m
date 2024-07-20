function [acc, nmi, Xn, Ls] = doSensitivityAnalysis(ds, noise, options)



%% Step 1: copy hyperparameters
expe_prt = options.expe_prt;% turn off print_info
P = options.nclients;
is_loadXmask = 0;% indicator variable for consistent alternative of partition of data

%% Step 2: load data
dispIteration('****** Data matrix is loading ...', expe_prt);
[X, Label] = fedsc_dataloader(ds.name(ds.id));
dispIteration('Loaded dataset: ' + ds.name(ds.id) + '_tsne.mat.', 1);
% extract some info
[m, n] = size(X);
L = Label;
K = length(unique(Label));% number of clusters
Lr = zeros(length(L), 2);% store cluster assignments of four methods; col 1 for vanilla Lr; col 2 for FedSC Lr


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
dispIteration('****** Sensitivity analysis starts ******', expe_prt);
% create variables err and nmi for restoring the metric info of clustering
% performance
acc = zeros(1, 2);% row: datasets; col: methods; col 1 for vanilla SC; col 2 for FedSC
nmi = zeros(1, 2);% row: datasets; col: methods; col 1 for vanilla SC; col 2 for FedSC

%% Method II: FedSC
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
[cellLr, cellKxx, ~, options] = FedSC(Xn, Xmask, K, options);
algKxx = cellKxx{1, 2};
Lr(:, 2) = cellLr{1, 2};
[acc(1, 2), nmi(1, 2)] = cluster_metrics(L, Lr(:, 2));
Lr(:, 2) = bestMap(L, Lr(:, 2));
% options = rmfield(options, ["d", "sigma2"]);


%% Method I: Vanilla SC
tKxx = kernel(Xn, Xn, options.sigma2);% ground truth of similarity matrix
% sparse tKxx
numKnn = max(ceil(log(n)), 1);
knn_type = 'complete';
knn_tKxx = getSimilarityMatrixFromKernelMatrix(tKxx, numKnn, knn_type);
% perform SC
Lr(:, 1) = SpectralClustering(knn_tKxx, K);
[acc(1, 1), nmi(1, 1)] = cluster_metrics(L, Lr(:, 1));
Lr(:, 1) = bestMap(L, Lr(:, 1));
dispIteration('Done.', expe_prt);


Ls = [L, Lr];

end