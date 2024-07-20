function plot_ClusterLr(X, Ls, Legends, show_mode)

if strcmp(show_mode, 'stack')
    figure;
    for i = 1:size(Ls, 2)
        subplot(1, 3, i);
        gscatter(X(:, 1), X(:, 2), Ls(:, i), 'rc');% proposed FedSC
        title(Legends(i));
    end
else
    for i = 1:size(Ls, 2)
        figure;
        gscatter(X(:, 1), X(:, 2), Ls(:, i), 'rc');% proposed FedSC
        %title(Legends(i));
    end
end

end