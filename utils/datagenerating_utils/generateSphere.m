function [X, Y, Z] = generateSphere(r, d)
% content: randomly generate sphere
% Input:
%       r: scalar, radius of sphere
%       d: scalar, number of uniformly sampling points
% Output:
%       X: X-coordinate meshgrid
%       Y: Y-coordinate meshgrid
%       Z: Z-coordinate meshgrid
%
% Author: D.Q.

[X, Y, Z] = generateEllipsoid([r, r, r], d);

end