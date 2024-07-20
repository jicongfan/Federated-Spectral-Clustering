function [X, Y, Z] = generateEllipsoid(r, d)
% content: randomly generate sphere
% Input:
%       r: 3 by 1, principal axes of ellipsoid
%       d: scalar, number of uniformly sampling points
% Output:
%       X: X-coordinate meshgrid
%       Y: Y-coordinate meshgrid
%       Z: Z-coordinate meshgrid
%
% Author: D.Q.
theta = (-pi:pi/d:0)';
phi = (0:2*pi/d:2*pi)';
ThetaLen = length(theta);
PhiLen = length(phi);

% X
if abs(r(1)) < 1e-4
    X = zeros(ThetaLen, PhiLen);
else
    sin_theta = sin(theta);
    X = r(1)*sin_theta*cos(phi)';
end

% Y
if abs(r(2)) < 1e-4
    Y = zeros(ThetaLen, PhiLen);
else
    Y = r(2)*sin_theta*sin(phi)';
end

% Z
if abs(r(3)) < 1e-4
    Z = zeros(ThetaLen, PhiLen);
else
    e = ones(PhiLen, 1);
    Z = r(3)*cos(theta)*e';
end

end