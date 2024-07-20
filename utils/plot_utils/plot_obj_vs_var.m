function flag = plot_obj_vs_var(obj_list, sigma_list)
% Content: plot the curve of obj with different variance of noisy matrix
%
%
%
try
    figure
    
    for i = 1: length(sigma_list)
    plot(obj_list{i, 1}, 'DisplayName', ['sigma = ', num2str(sigma_list(i))]);
    hold on;
    end
    legend;
    ylabel('obj');
    xlabel('sigma');
    %title('Value of objective function vs. Variance of noisy matrix');
    
    flag = 0;
catch
    flag = -1;
end

end