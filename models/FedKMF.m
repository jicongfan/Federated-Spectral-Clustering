function [Z, C, obj, options] = FedKMF(X, Xmask, options)
%%
% Federated spectral clustering based on gradient average
% Input
%       X:        the data matrix, mxn, m featrures and n samples
%       d:        dimension of the dictionary Z
%       lambda:     penalty parameter for C
%       options:
%               options.rbf_c:    a constant for estimating the sigma of Gaussian kernel,
%                                 e.g. 1, 3,or 5. (default 1)
%               options.tol:      tollorence of stopping (default 1e-4)
%               options.maxiter:  maximum iterations (default 300)
% output
%       Xr:       recovered matrix
%       Z:        dictionary
%       C:        coefficient matrix
%       options:  parameter sets
%
%

[m, n] = size(X);
prt = options.prt;
%% check hyperparameters
if isfield(options,'rbf_c')
    % check hyperparameter of gauss kernel
    rbf_c=options.rbf_c;
else
	rbf_c=1;
end

dispIteration('****** Computing the sigma2 of Gaussian kernel ......', prt);

if ~isfield(options, 'sigma2')
    sigma2 = sigma2_estimator(X, rbf_c);
    % XX = sum(Xs.*Xs,1);
    % dist = repmat(XX,size(Xs,2),1) + repmat(XX',1,size(Xs,2)) - 2*(Xs'*Xs);
    % sigma2 = (mean(real(dist(:).^0.5))*rbf_c)^2;
    dispIteration(['****** The estimated sigma2 of algKxx is ' num2str(sigma2)], 1);
else
    sigma2 = options.sigma2;
    dispIteration(['****** The predefined sigma2 is ' num2str(sigma2)], 1);
end

if ~isfield(options, 'd')% isempty(d)
    d = d_estimator(X, sigma2, options.d_tol);
    % Kxx = kernel(Xs, Xs, sigma2);
    % [~,S,~] = svd(Kxx);
    % S = diag(S);
    % figure
    % S = cumsum(S)/sum(S);
    % bar(S)
    % d = find(S > options.d_tol, 1);
    % d = find(S > 0.8, 1);
    dispIteration(['****** The svd-based estimated d is ' num2str(d)], 1);
else
    d = options.d;
    dispIteration(['****** The predefined d is ' num2str(d)], 1);
end

clear XX dist Xs Kxx U S V

if isfield(options,'tol')
    tol = options.tol;
else
	tol=1e-4;
end
%
if isfield(options,'maxiter')
    maxiter = options.maxiter;
else
	maxiter = 50;
end
if isfield(options,'compute_obj')
    compute_obj = options.compute_obj;
else
	compute_obj = 0;
end
%
if isfield(options,'eta')
    v_eta = options.eta;
else
	v_eta = 0.5;
end

if isfield(options, 'nclients')
    nclients = options.nclients;
else
    nclients = n;
end

if isfield(options, 'S')
    S = options.S;
else
    S = 300;
end

% if isfield(options, 'm')
%     m = options.m;
% else
%     m = ceil(0.75*n);
% end

%% save options
options.tol = tol;
options.maxiter = maxiter;
options.sigma2 = sigma2;
options.eta = v_eta;
options.d = d;
lambda = options.lambda;
% options

%% Setup for Fedederated Space Factorization
% Initialize the matrix Z_1^0, Z_2^0, ..., Z_P^0 at the server side,
% initialize the matrix \{C_p^0\}_{p = 1}^P at the clients, 
% where A^0 = \{1, ..., P\} is a partition of clients
% Hyperparameter Q_hat

dispIteration('****** Initialize the matrix \{Z_p^0\}_{p = 1}^P at the server side...', prt);
Zs = cell(nclients, 1);
for i = 1: nclients
    Zs{i, 1} = randn(m, d);
end

dispIteration('****** Initialize the matrix \{C_p^0\}_{p = 1}^P at the clients...', prt);
C = zeros(d, n);
C_new = zeros(d, n);

dispIteration('****** Initialize the preset parameter \hat{Q}...', prt)
% Federated alternating optimization
Z = zeros(m, d);
A = 1:nclients;% set of clients participating FedSC
%
if n<18000
    Kxx = eye(n);
end
J =inf;
obj=[];

%% federated learning starts ...
dispIteration('****** Federated alternating optimization starts...', prt);
for s = 1: S
    if s < S && mod(s, 50) ~= 0
        fprintf('.');
    else
        fprintf('.\n');
    end
    
    if s < 5
        eta = 0;
    else
        eta = v_eta;
    end
    
    dispIteration(' ', prt);
    dispIteration('-----------------------------------------------------------------', prt);
    dispIteration(['Epoch ', num2str(s), ': Federated Spectral Clustering'], prt);
    dispIteration(' ', prt);
    dispIteration('++++++ Server-side update ---> Parameter Average ++++++', prt);
    dispIteration('****** Update the matrix Z...', prt);
    Z_new = zeros(m, d);
    for i = A
        if options.is_perturbed_by_factors
            % inject gaussian noise into Z and C
            % dispIteration('****** Gaussian noise is being injected into Z_p before it is posted after each local update', 1);
            fprintf('-');
            Zs{i, 1} = addNoise2X(Zs{i, 1}, options.z_noise_ratio);
        end
        Z_new = Z_new + Zs{i, 1};
    end
    Z_new = Z_new/length(A);
    
    dispIteration('****** Evaluate the objective function......', prt);
    Kzx=kernel(Z_new,X,sigma2);
    Kzz=kernel(Z_new,Z_new,sigma2);
    [obj, J_new] = dispLogInfo(s, obj, compute_obj, J, C, C_new, Z, Z_new, Kxx, Kzx, Kzz, lambda, prt);
    if s > 200 && var(obj(end - 7: end)) < 1e-4
        fprintf('\n');
        dispIteration('****** FedSC converged with the variance of less than 10 of the 20 latest obj values ......', prt);
        break;
    end
    C = C_new;
    Z = Z_new;
    try
        J = J_new;
    catch
        disp('Terminated by too big noise.');
        break;
    end
    
    dispIteration('****** Server-side parameter aggregation finished', prt);
    % dispIteration(['****** Select a set of clients A^s (with size |A^s| = ', num2str(sampling_m), ') based on the sample noise - ', Sampling_noise], prt);
    % A = samplingClients(sampling_m, PMF, 'nonrep');% sampling clients without replacement
    dispIteration('****** Brodcast Z^s to all clients', prt);
    
    dispIteration(' ', prt);
    
    %% client
    dispIteration('++++++ Client-side update ---> Local optimization ++++++', prt);
    for p = 1: nclients
        dispIteration(' ', prt);
        dispIteration(['****** Client ', num2str(p), ': local update start...'], prt);
        dispIteration(['****** Client ', num2str(p), ' starts to receive the global matrix Z...'], prt);
        Zs{p, 1} = Z;
        
        Kzx=kernel(Zs{p, 1},X(:, Xmask{p, 1}),sigma2);
        Kzz=kernel(Zs{p, 1},Zs{p, 1},sigma2);
        dispIteration(['****** Client ', num2str(p), ' starts to update the local matrix C...'], prt);
        C_new(:, Xmask{p, 1}) = localUpdateC(Kzx, Kzz, d, lambda);

        dispIteration(['****** Client ', num2str(p), ' starts to update the local matrix Z...'], prt);
        Zs{p, 1} = localUpdateZ(X(:, Xmask{p, 1}), Zs{p, 1}, Kzx, Kzz, C_new(:, Xmask{p, 1}), sigma2, eta);

        dispIteration(['****** Denote Z_', num2str(p), '^', num2str(s), ' = Z_', num2str(p), '^{', num2str(s), ', Q^', num2str(s), '} and C_', num2str(p), '^', num2str(s), ' = C_', num2str(p), '^{', num2str(s), ', Q^', num2str(s), '}'], prt);
        if ismember(p, A) == 1
            dispIteration(['****** Upload Z_', num2str(p), '^', num2str(s), ' to the server'], prt);
        end
    end
end

% add noise to the coefficient matrix C_p of the client p with N_p data points for p = 1, 2,
% \dots, P

% add noise pattern
% sigma = std(X, 1, 'all');% 1 represents
% E = gamma*sigma*randn(size(X));% Gaussian noise e_{ij} \sim \mathcal{N}(0, \sigma^2)
% Xr = X + E;
% end


% if options.is_perturbed_by_factor
%     % inject gaussian noise into Z and C
%     dispIteration('****** Gaussian noise is being injected into Z ...', prt);
%     Zn = addNoise2X(Z, options.c_noise_ratio);
% 
%     dispIteration('****** Gaussian noise is being injected into C at final step', prt);
%     Cn = addNoise2X(C, options.z_noise_ratio);
% else
%     Cn = C;
%     Zn = Z;
% end

if options.is_perturbed_by_factors
    dispIteration('****** Gaussian noise is being injected into C at final step', 1);
    % inject noise to the coefficient matrix C for differential privacy of
    % FedSC
    % C \in \mathbb{R}^{d \times n}
    % Xmask{p, 1} \in \mathbb{Z}^{1 \times N_p}
    
    for p = 1: nclients
        C(:, Xmask{p, 1}) = addNoise2X(C(:, Xmask{p, 1}), options.c_noise_ratio);
    end

    dispIteration('Security-Enhanced FedSC: Action of noise injection into C completed', prt);
end

dispIteration('Federated similarity reconstruction completed', 1);

end

%% group of utils: update schemes of local variables C_p^s and Z_p^{s, t}
function C_new = localUpdateC(Kzx, Kzz, d, lambda)
% content: use gradient method to solve the local objective function:
% f_p(Z^{s, 0}, C_p^s) = \frac{1}{2}\|\phi(X_p) - \phi(Z^{s, 0})C_p^s\|_F^2 + \frac{\lambda_C}{2}\|C_p\|_F^2

C_new= (Kzz+lambda*eye(d))\Kzx;

end

%
function Z_new = localUpdateZ(X, Z, Kzx, Kzz, C_new, sigma2, eta)
% content: use gradient method to solve the local objective function:
% f_p(Z^{s, 0}, C_p^s) = \frac{1}{2}\|\phi(X_p) - \phi(Z^{s, 0})C_p^s\|_F^2 + \frac{\lambda_C}{2}\|C_p\|_F^2

vZ = 0;
g_Kxz = -C_new';
g_Kzz = 0.5*(C_new*C_new');
[g_Z1,~,C1] = gXY(g_Kxz,Kzx',X,Z,sigma2,'Y');
[g_Z2,T2,C2] = gXX(g_Kzz,Kzz,Z,sigma2);
tau = 1/sigma2*(2*T2-diag(C1(1,:)+2*C2(1,:)));
% tau=normest(tau);
g_Z=(g_Z1+g_Z2)/tau;
vZ=eta*vZ+g_Z;
Z_new=Z-vZ;

end

%% group of utils for local update of Z_p^{s, t}
function [g,T,C]=gXY(g_Kxz,Kxz,X,Z,sigma2,v)
switch v
    case 'Y'
        T=g_Kxz.*Kxz;% n x d
        C=repmat(sum(T),size(X,1),1);
        g=1/sigma2*(X*T-Z.*C);  
    case 'X'
        T=g_Kxz'.*Kxz';% d x n;
        C=repmat(sum(T),size(X,1),1);
        g=1/sigma2*(Z*T-X.*C);
end
end

%
function [g,T,C]=gXX(g_Kzz,Kzz,Z,sigma2,I)
if ~exist('I')
    T=g_Kzz.*Kzz;
    C=repmat(sum(T),size(Z,1),1);
    g=2/sigma2*(Z*T-Z.*C);
else
    T=g_Kzz.*Kzz;
    C=repmat(sum(T),size(Z,1),1);
    g=2/sigma2*(Z.*repmat(diag(T)',size(Z,1),1)-Z.*C);
end
end 


%% group of utils: print info
function [obj, J_new] = dispLogInfo(iter, obj, compute_obj, J, C, C_new, Z, Z_new, Kxx, Kzx, Kzz, lambda, prt)
%
formatSpec = '%.2e';
%
if size(Z, 1) < 18000
    J_new=0.5*trace(Kxx-C'*Kzx-Kzx'*C+C'*Kzz*C)+0.5*lambda*sum(C(:).^2);
    obj(iter)=J_new;
    dJ=(J-J_new)/J;
    J=J_new;
else
    J='NotComputed';
    dJ='NotComputed';
end
%
if (iter<10||mod(iter,50)==0||compute_obj) && (prt == 1)
    dC=norm(C-C_new,'fro')/norm(C,'fro');
    dZ=norm(Z-Z_new,'fro')/norm(Z,'fro');
    disp(['Epoch ' num2str(iter) ': [J dJ dC dZ] = ['...
            num2str(J, formatSpec) ' ' num2str(dJ, formatSpec) ' '...
            num2str(dC, formatSpec) ' ' num2str(dZ, formatSpec) ']']);
end
%     if max([dC,dZ])<tol||dJ<tol
%         disp('Converged!')
%         break
%     end
end
