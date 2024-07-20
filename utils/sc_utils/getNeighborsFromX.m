function [e, global_rows, local_rows] = getNeighborsFromX(X, L, radius)
% Content: get the different kinds of neighbors of each node after
% doing the spectral clustering on X.
% Input:
%       X: n by d, design matrix
%       L: n by 1, ground truth of array of labels
%       Lr: n by 1, predictive array of labels
%
% Output:
%       e: scalar, average of {inta neighbors/inter neighbors} raios of each node in X
%       neighbors: n by 2, cell for the storage of different kinds of
%       neighbors of each node in X.
%

n = size(X, 1);
lbls = unique(L);

% get the global neighbors of each node
global_rows = rangesearch(X, X, radius);


% get the intra-class neighbors of each node
local_rows = cell(n, 1);
for i = 1: length(lbls)
    % figure out the intra-class neighbors of each node with the label of L(i)
    mask = L == lbls(i);
    tmp_rows = rangesearch(X(mask, :), X(mask, :), radius);
    mask = find(mask == 1);
    for j = 1: length(mask)
        local_rows{mask(j)} = tmp_rows{j};
    end
end

% compute the average of local/glocal-neighbor ratio for X
e = zeros(n, 1);
for i = 1: n
    e(i, 1) = numel(intersect(global_rows{i, 1}, local_rows{i, 1}))/numel(union(global_rows{i, 1}, local_rows{i, 1}));
end
%e = mean(e);

end