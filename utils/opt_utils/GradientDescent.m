function [x, iter] = myGradient(func, x0, tol, maxit, varargin)
%%
% Content: myGradient uses gradient descent with Armijo back-tracking line search for solving the minimum of func
% Parameter List of Input:
% func:function to be optimized
% x0:initial guess
% tol: tolerance value for termination
% maxit: maximum number of iterations allowed
% varargin: possible parameters required by the func
%
% Parameter List of Output:
% x: stationary point of func
% iter: number of iteration before arriving at stationary point
%
% Version: Sep 29, 2021
% Written by Dong Qiao whose email address is dongqiao@link.cuhk.edu.cn
%%

x = x0;
iter = 1;

[f0, g0] = feval(func, x0, varargin{:});
gnrm0 = norm(g0);
g = g0;

% parameters for line search
alpha = 1e-3; % initial step
C = f0;
Q = 1;
gamma = 0.85;
while iter<=maxit
    % Armijo back-tracking line search
    d = - g;
    alpha = bls(alpha, func, x, d, C, varargin{:});
    
    % update x
    x_new = x + alpha*d;
    
    % update g_new
    [f_new, g_new] = feval(func, x_new, varargin{:});
    
    % print info of iteration
    gnrm = norm(g_new);
    if mod(iter, 50) == 0
        disp(['iter   ', num2str(iter), ':   gnrm/gnrm0 = ', num2str(gnrm/gnrm0)]);
    end
    
    % check ||gnrm|| < tol*||gnrm0||
    if gnrm <= tol*gnrm0
        % print info of iteration when converged
        disp(['iter   ', num2str(iter), ':   gnrm/gnrm0 = ', num2str(gnrm/gnrm0)]);
        break;
    end
    
    % BB step
    s = x_new - x;
    y = g_new - g;
    sy = s'*y; ss = s'*s; yy = y'*y;
    if ~(sy == 0)
        if mod(iter, 2) == 0
            alpha = ss/sy;
        else
            alpha = sy/yy;
        end
    end
    
    
    % update the cofficient C for f_k in line search (Zhang & Hager)
    Q_new = gamma * Q + 1;
    C = (gamma*Q*C + f_new) / Q_new;
    
    x = x_new;
    g = g_new;
    Q = Q_new;
    
    % iter ++
    iter = iter + 1;
end

end

function alpha = bls(alpha0, func, x_k, d_k, C, varargin)
% Content: Armijo back-tracking line search is used to compute the proper step size
% Parameter List of Input:
% alpha0: initial step size
% func:function to be optimized
% x_k: x under the iteration of k
% d_k: direction under the iteration of k
% C: recurrence variable of f under the iteration of k
% varargin{:}: optional parameter list of func
%
% Parameter List of Output:
% lambda: step size under the iteration of k
% 
% Version: Sep 29, 2021
% Written by Dong Qiao whose email address is dongqiao@link.cuhk.edu.cn

rho = 1e-4; 
alpha = alpha0; sigma = 0.75;
m = 0; mmax = 10;
while m <= mmax,
    f_new = feval(func, x_k+alpha*d_k, varargin{:});
    if (f_new <= C + rho*alpha*norm(d_k)^2)
        break;
    else
        alpha = sigma * alpha;
        m = m + 1;
    end
end

end
