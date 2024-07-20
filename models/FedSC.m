function [cellLr, cellKxx, obj, options] = FedSC(X, Xmask, K, options)
% Content: performe federated spectral clustering (FedSC)
% Input:
%       X: m by n, data matrix with n data points of m features
%       K: N by 1, number of clusters
%       Xmask: N by 1, cell of data partition
%       options: struct, set of hyperparameters
%
% Output:
%       Lr: N by 1, predictive result of cluster assignments
%
%

% check parameters
if ~isfield(options, 'attacker')
    options.attacker = 0;
end

prt = options.prt;
dispIteration('****** Method IV: FedSC ******', prt);
% Stage I: Federated Similarity Reconstruction
dispIteration('****** Stage I: Federated Similarity Reconstruction ...', prt);
[Z, C, obj, options] = FedKMF(X, Xmask, options);
options.tau_C = opnrm(C, '2infty');
Kxz = kernel(X, Z, options.sigma2);
options.opt_gamma = max(diag(C'*kernel(Z, Z, options.sigma2)*C - C'*Kxz' - Kxz*C + kernel(X, X, options.sigma2)));

% Stage II: Spectral Clustering
n = size(C,2);
% disp('Constructing affinity matrix ...');

% step 1: compute pairwise distance matrix
Kzz = kernel(Z, Z, options.sigma2);
algKxx = C'*Kzz*C;
algKxx = reshape(mapminmax(algKxx(:)', 0, 1), size(algKxx));
% algKxx = algKxx - diag(diag(algKxx));
cellKxx{1, 1} = algKxx;

% step 2: compute similarity matrix
if isfield(options, 'numKnn')
    numKnn = options.numKnn;
    dispIteration(['****** The predefined numKnn is ' num2str(numKnn)], prt);
else
    numKnn = max(ceil(log(n)), 1);
    dispIteration(['****** The paper-based estimated numKnn is ' num2str(numKnn)], prt);
end
knn_type = 'complete';

% I = algKxx <= 0.72;
% for j = 1: size(algKxx, 2)
%     algKxx(I(:, j), j) = 0;
% end

knn_algKxx = getSimilarityMatrixFromKernelMatrix(algKxx, numKnn, knn_type);
% Kxx = filterSimilarityMatrix(Kxx, 6);
cellKxx{1, 2} = knn_algKxx;

% step 3: do spectral clustering
fprintf('\n');
dispIteration('****** Stage II: Federated Spectral Clustering ...', prt);
cellLr{1, 1}= 1;%SpectralClustering(algKxx, K);
cellLr{1, 2}= SpectralClustering(knn_algKxx, K);

if options.attacker
    dispIteration('****** Stage II: Federated Spectral Clustering - Attacker ...', prt);
    % C: d \times n
    cellLr{1, 3}= kmeans(C', K, 'Replicates', options.replicates);
    % Lr = spectralcluster(algKxx, K);
    dispIteration('****** Secured Spectral Clustering completed ...', prt);
end

end
