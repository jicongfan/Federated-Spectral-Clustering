function v = opnrm(X, p)

v = 0;
if strcmp(p, "2infty")
    % maximum of the \ell_2 norms of the matrix columns
    for i = 1:size(X, 2)
        nrm = norm(X(:,i));
        if nrm > v
            v = nrm;
        end
    end
else
    fprintf('Error: Illegal hyperparameter - p');
end


end