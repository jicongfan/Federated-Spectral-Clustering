function [K,XY]=kernel(X, Y, sigma2)
% Content: gauss kernel function
% Formula: k(x_i, x_j) = \exp(-\|x_i - x_j\|_2^2/(2*sigma^2))
% Input:
%       X: m by n_1
%       Y: m by n_2
%       kscale: squared sigma (bandwidth)
%
% Output:
%       K: n_1 by n_2, kernel matrix
%       XY:
% 
nx=size(X,2);
ny=size(Y,2);
XY=X'*Y;
xx=sum(X.*X,1);
yy=sum(Y.*Y,1);
D=repmat(xx',1,ny) + repmat(yy,nx,1) - 2*XY;
K=exp(-D/2/sigma2); 
end