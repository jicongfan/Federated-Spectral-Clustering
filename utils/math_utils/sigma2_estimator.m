function sigma2 = sigma2_estimator(X, rbf_c)
% content: estimate the hyperparameter sigma^2 for gaussian kernel: exp(-\|x_1 - x_2\|_2^2/(2sigma^2))
% Args
% Xs: m by n, real, data matrix with m features and n data points
%
% Output
% sigma2: scalar, estimated sigma2


n = size(X, 2);
if n<8000
    Xs = X;
else
    Xs = X(:,randperm(n,8000));
end

XX = sum(Xs.*Xs,1);
dist = repmat(XX,size(Xs,2),1) + repmat(XX',1,size(Xs,2)) - 2*(Xs'*Xs);
sigma2 = (mean(real(dist(:).^0.5))*rbf_c)^2;

end