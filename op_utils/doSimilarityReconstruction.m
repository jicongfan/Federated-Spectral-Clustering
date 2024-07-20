function [acc, nmi, relKxx, obj, cellKxx, L, Xs] = doSimilarityReconstruction(ds, noise, options)


% check parameter
if ~isfield(options, 'attacker')
    options.attacker = 0;
end

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

%% Step 3: inject noise
if noise.gammas(noise.xid) > 0
    dispIteration('****** Gaussian noise is being injected into data matrix ...', expe_prt);
    dispIteration(['****** Noise_gamma = ', num2str(noise.gammas(noise.xid))], expe_prt);
    Xn = addNoise2X(X, noise.gammas(noise.xid));
    noise_sigma = noise.gammas(noise.xid)*std(X, 1, 'all');
    dispIteration('****** Done.', expe_prt);
else
    Xn = X;
    noise_sigma = 0;
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
dispIteration('****** Federated similarity reconstruction starts ******', expe_prt);
% Step 5.1: create variables
% create variables err and nmi for restoring the metric info of clustering
% performance
if options.attacker
    acc = zeros(1, 2);% row: datasets; col 1 for standard; col 2 for attacker
    nmi = zeros(1, 2);% row: datasets; col 1 for standard; col 2 for attacker
    Lr = zeros(length(L), 2);% store cluster assignments of four methods; col 1 for Lr; col 2 for Lr-attacker
else
    acc = 0;% row: datasets; col 1 for standard; col 2 for attacker
    nmi = 0;% row: datasets; col 1 for standard; col 2 for attacker
    Lr = zeros(length(L), 1);% store cluster assignments of four methods; col 1 for Lr; col 2 for Lr-attacker
end
relKxx = zeros(1, 3); % col 1 for the reconstruction error between algKxx and tKxx; col 2 for the reconstruction error between knn_algKxx and knn_tKxx; col 3 for the upper bound on reconstruction error
% Step 5.2: inject noise
if noise.flag_zc
    % turn on the mode: perturb factors
    options.is_perturbed_by_factors = 1;

    dispIteration('****** Achieving DP-FedSC by perturbing Z and C ...', 1);
    options.z_noise_ratio = noise.gammas(noise.zid);
    dispIteration(['****** options.Z_Noise_gamma = ', num2str(options.z_noise_ratio)], expe_prt);
    
    options.c_noise_ratio = noise.gammas(noise.cid);
    dispIteration(['****** options.C_Noise_gamma = ', num2str(options.c_noise_ratio)], expe_prt);
end

% Step 5.3: perform FedSC for P = 8
[cellLr, cellKxx, obj, options] = FedSC(Xn, Xmask, K, options);
Lr(:, 1) = cellLr{1, 2};
% Lr(:, 4) = bestMap(L,Lr(:, 4));
% [acc(ds_id, 4), nmi(ds_id, 4)] = cluster_metrics(L, Lr(:, 4));
[acc(1, 1), nmi(1, 1)] = cluster_metrics(L, Lr(:, 1));

% Step 5.4: compute reconstruction error
% Step 5.4.1: tKxx and knn_tKxx
% construct kernel matrix
% tKxx = kernel(Xn, Xn, options.sigma2);% ground truth of similarity matrix with identical hyperparameter of kernel function
sigma2 = sigma2_estimator(X, options.rbf_c);
fprintf('****** The estimated sigma2 of tKxx is %.4f\n', sigma2);
tKxx = kernel(X, X, sigma2);% ground truth of similarity matrix with identical hyperparameter of kernel function
% clear sigma2;
cellKxx{1, 3} = tKxx;

% make it sparse based on KNN operation
numKnn = max(ceil(log(n)), 1);
knn_type = 'complete';
knn_tKxx = getSimilarityMatrixFromKernelMatrix(tKxx, numKnn, knn_type);
cellKxx{1, 4} = knn_tKxx;
% Step 5.4.2: cellKxx{1, 1} for algKxx; cellKxx{1, 2} for knn_algKxx;
% Step 5.4.3: calculate
% formulas
relKxx(1, 1) = norm(cellKxx{1, 1} - tKxx, 'fro')/norm(tKxx, 'fro');
% relKxx(1, 1) = norm(cellKxx{1, 1} - tKxx, 'fro');
% relKxx(1, 2) = norm(cellKxx{1, 2} - knn_tKxx, 'fro')/norm(knn_tKxx, 'fro');


if options.attacker
    Lr(:, 2) = cellLr{1, 3};% attacker
    [acc(1, 2), nmi(1, 2)] = cluster_metrics(L, Lr(:, 2));% attacker
    % options = rmfield(options, ["d", "sigma2"]);
end

% Step 5.4.4: compute the upper bound on reconstruction error
t = 0.99;
xi = sqrt(m + 2*sqrt(m*t) + 2*t);
theta = opnrm(X, '2infty');
tau_C = options.tau_C;
options = rmfield(options,'tau_C');
opt_gamma = options.opt_gamma;% \|\phi(Z)C - \phi(\hat{X})\|_{2,\infty} \le opt_gamma
relKxx(1, 3) = ((noise_sigma*xi+sqrt(2)*theta)^2 - 2*theta^2)/sigma2 + (sqrt(options.d)*tau_C+ 1)*opt_gamma;

% Step 5.4.5: store raw X and noisy Xn
Xs{1,1} = X;
Xs{1,2} = Xn;

end