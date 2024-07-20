function Xr = addNoise2X(X, gamma)
% Content:
% Input
%       X: m by n, data matrix
%
% Output
%       Xr: m by n, data matrix perturbed by noise
%

% note that we do not give random seed here. Instead, we will reimplement
% the experiment of comparative analysis for some times and show the final
% result as the average of multiple acc and NMI with their
% variance.
% rng(0);
sigma = std(X, 1, 'all');% 1 represents
E = gamma*sigma*randn(size(X));% Gaussian noise e_{ij} \sim \mathcal{N}(0, \sigma^2)
Xr = X + E;


end