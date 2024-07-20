function d = d_estimator(X, sigma2, d_tol)
% content: estimate the hyperparameter d for matrix factorization: X = Z*C
% where X \in \mathbb{R}^{m \times n}, Z \in \mathbb{R}^{m \times d}, C \in
% \mathbb{R}^{d \times n}
% Args
% X: m by n, real, data matrix with m features and n data points
%
% Output
% d: scalar, estimated d


n = size(X, 2);
if n<8000
    Xs = X;
else
    Xs = X(:,randperm(n,8000));
end

Kxx = kernel(Xs, Xs, sigma2);
[~,S,~] = svd(Kxx);
S = diag(S);
% figure
S = cumsum(S)/sum(S);
% bar(S)
d = find(S > d_tol, 1);

end