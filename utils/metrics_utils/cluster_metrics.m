function [e, nmi] = cluster_metrics(L, Lr)
% Content: calculate the score of clustering result
% Input:
%       L: N by 1, ground truth of cluster assignments
%       Lr: N by 1, predictive results
%
% Output:
%       e: scalar, misclassification rate
%       nmi: scalar, NMI
%

Lr = bestMap(L,Lr);

e = 1 - calMisclassificationRate(L, Lr);
% e = calMisclassificationRate(L, Lr);
nmi = calNMI(L, Lr);

end

function s = calMisclassificationRate(L, Lr)
% Content: calculate the misclassification rate of clustering result
% Input:
%       L: N by 1, ground truth of cluster assignments
%       Lr: N by 1, predictive results
%
% Output:
%       s: scalar, score of clustering performance
%
%

s = sum(L(:) ~= Lr(:)) / length(L);

end

function s = calNMI(L, Lr)
% Content: The function is to calculate NMI (normalized mutual information) metric.
% Input:
%       L: N by 1, ground truth of cluster assignments
%       Lr: N by 1, predictive results
%
% Output:
%       s: scalar, score of clustering performance
%
% References:
% [1] Ruggero G. Bettinardi (2022). getNMI(A,B) 
%     (https://www.mathworks.com/matlabcentral/fileexchange/62974-getnmi-a-b), 
%     MATLAB Central File Exchange. Retrieved October 9, 2022.

s = getNMI(L, Lr);

end