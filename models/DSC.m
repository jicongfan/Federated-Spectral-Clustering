function Lr = DSC(X, Xmask, K, options)
% Content: iterative federated clustering algorithm (ICFA)
% Input:
%   X: m by n, data matrix with m samples of n features
%   K: integer, number of clusters
%
% Output:
%
%
%
% References:
% [1] A. Ghosh, J. Chung, D. Yin, and K. Ramchandran. An efficient framework 
%     for clustered federated learning. Advances in Neural Information 
%     Processing Systems, 33:19586–19597, 2020.
%

[m, ~] = size(X);

% copy hyperparameters
P = size(Xmask, 1);
kmeans_replicates = options.replicates;

% select an arbitary node to start the process, let's assume node 1 for
% simplicity.
% In node 1, compute V_1, centroids_1 = SpectralClustering(X_1)
pid = randi(P);
V = zeros(m, K);
% [~, V(Xmask{pid, 1}, :)] = spectralcluster(X(Xmask{pid, 1}, :), K);
Kxx = internal.stats.similarity(X(Xmask{pid, 1}, :));
V(Xmask{pid, 1}, :) = internal.stats.spectraleigs(Kxx, K);
centroids = zeros(P*K, K);
[~, centroids(K*(pid - 1) + 1: K*pid, :)] = kmeans(abs(V(Xmask{pid, 1}, :)), K, 'replicates', kmeans_replicates);

% From node 1 send to the other nodes a randomly selected data point $d \in
% D_1$ and its eigenvector $v_1^d \in V_1$
xid = randi(length(Xmask{pid, 1}));
d = X(Xmask{pid, 1}(1, xid), :);
vd = V(Xmask{pid, 1}(1, xid), :);

for i = 1: P
    if i == pid
        continue
    end
    
    % X_i = X_i \cap d
    Xp = [X(Xmask{i, 1}, :); d];
    % compute V_i = eigenvectors(X_i)
    % [~, Vp] = spectralcluster(Xp, K);
    Xp = internal.stats.similarity(Xp);
    Vp = internal.stats.spectraleigs(Xp, K);
    % get eigenvector v_i^d associated d
    Vp_vd = Vp(end, :);
    for c = 1: K
        if sign(Vp_vd(c)) ~= sign(vd(c))
            Vp(:, c) = - Vp(:, c);%sign flip
        end
    end
    V(Xmask{i, 1}, :) = Vp(1: end - 1, :);
    [~, centroids(K*(i - 1) + 1: K*i, :)] = kmeans(abs(V(Xmask{i, 1}, :)), K, 'replicates', kmeans_replicates);
    % send centroids_p to the selected node
end

% In node 1, compute centroids = kmeans({centroids_1, centroids_2, ..., centroids_P}, K)
% assemble all centroids into one centroid
[~, C] = kmeans(centroids, K, 'replicates', kmeans_replicates);

% Return centroids as the global centroids
Lr = zeros(m, 1);
for i = 1: m
    [~, Lr(i, 1)] = min(sum((repmat(V(i, :), K, 1) - C).^2, 2));
end

end