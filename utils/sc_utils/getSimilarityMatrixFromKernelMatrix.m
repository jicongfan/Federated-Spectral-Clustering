function S = getSimilarityMatrixFromKernelMatrix(Kxx, numKnn, knn_type)
% Content: calculate the similarity matrix
% Input:
%       D: n by n, distance matrix where each element d_{ij} represents the
%       distance between node i and node j.
%       numKnn: scalar, number of the nearest neighbors to be considered.
%       knn_type: 
%
% Output:
%       S: n by n, similarity matrix where each element s_{ij} represents
%       the similarity between node i and node j.
%

n = size(Kxx, 1);
[B, I] = sort(Kxx, 2, 'descend');
% neighbors = I(:, 1: numKnn);
rows = cell(n, 1);
weights = cell(n, 1);

% format the sorted data and index matrix to the required style.
for i = 1: n
    rows{i, 1} = I(i, 1: numKnn);
    weights{i, 1} = B(i, 1: numKnn);
end

S = makeSimilarityMatrixFromIndices(rows, weights, n);

% The KNN graph is not guaranteed to be symmetric. Make the matrix
% symmetric using either 'complete' or 'mutual'
if strcmp(knn_type, 'complete')
    S = max(S,S');
else % mutual
    S = min(S,S');
end

end

function S = makeSimilarityMatrixFromIndices(rows, weights, n)
% Content:
% Input:
%
%
% Output:
%
%
% Note 1: rows and weights are always cells (rangesearch and knnsearch-with-ties)
% Note 2: This function was built based on the backbone of
% internal.stats.similarity(). Please run `open internal.stats.similarity`
% in the command Window if you want to check the original function. Then,
% you will see the script of this function.

% Get column indices
weights = [weights{:}];
cols = ones(size(weights));
colind = 1;
for idx = 1: n
    num = length(rows{idx});
    cols(colind:colind+num-1) = idx*cols(colind:colind+num-1);
    colind = colind+num;
end
rows = [rows{:}];

% Return a sparse similarity matrix. 'sparse' does not accept weights with
% single precision. Convert weights to double.
S = sparse(rows, cols, double(weights), n, n);
end