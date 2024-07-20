function flag = plot_eFedMF_vs_var(eFedMF_list, sigma_list)
% Content: plot the curve of eFedRNLMF with different variance of noisy matrix
%
%
%
try
    figure
    
    plot(obj_list{i, 1}, 'DisplayName', ['sigma = ', num2str(sigma_list(i))]);
    ylabel('Error rate of misclassifying');
    xlabel('sigma');
    %title('Value of objective function vs. Variance of noisy matrix');
    
    flag = 0;
catch
    flag = -1;
end

end