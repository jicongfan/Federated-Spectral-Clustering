function [X, Z, C, options, obj] = FedKMF_fast(X, d, lambda, options)
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

%% check hyperparameters
if isfield(options,'rbf_c')
    % check hyperparameter of gauss kernel
    rbf_c=options.rbf_c;
else
	rbf_c=1;
end

disp('****** Computing the sigma2 of Gaussian kernel ......');
if n<8000
    Xs = X;
else
    Xs = X(:,randperm(n,8000));
end
XX = sum(Xs.*Xs,1);
dist = repmat(XX,size(Xs,2),1) + repmat(XX',1,size(Xs,2)) - 2*(Xs'*Xs);
sigma2 = (mean(real(dist(:).^0.5))*rbf_c)^2;
disp(['****** sigma2 = ' num2str(sigma2)]);

if isempty(d)
    Kxx = kernel(Xs, Xs, sigma2);
    [~,S,~] = svd(Kxx);
    S = diag(S);
    figure
    S = cumsum(S)/sum(S);
    bar(S)
    d = find(S>0.99,1);
    disp(['****** The estimated d is ' num2str(d)]);
end
clear XX dist Xs Kxx U S V

if isfield(options,'tol')
    tol=options.tol;
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
options.lambda = lambda;
options.eta = v_eta;
% options

%% Setup for federated learning
% Initialize the matrix Z_1^0, Z_2^0, ..., Z_P^0 at the server side,
% initialize the matrix \{C_p^0\}_{p = 1}^P at the clients, 
% where A^0 = \{1, ..., P\} is a partition of clients
% Hyperparameter Q_hat

disp('****** Partition all data points into n clients...');
Xmask = splitData2Clients(n, nclients);

disp('****** Initialize the matrix \{Z_p^0\}_{p = 1}^P at the server side...');
Zs = cell(nclients, 1);
for i = 1: nclients
    Zs{i, 1} = randn(m, d);
end

disp('****** Initialize the matrix \{C_p^0\}_{p = 1}^P at the clients...');
C = zeros(d, n);
C_new = zeros(d, n);

disp('****** Initialize the preset parameter \hat{Q}...')
% Federated alternating optimization
Z = zeros(m, d);
A = 1:nclients;% set of clients participating FedSC
%
if n<18000
    Kxx = eye(n);
end
J =inf;
obj=[];
prt = 1;

%% federated learning starts ...
disp('****** Federated alternating optimization starts...');
for s = 1: S
    if s < 5
        eta = 0;
    else
        eta = v_eta;
    end
    
    disp(' ');
    disp('-----------------------------------------------------------------');
    disp(['Epoch ', num2str(s), ': Federated Spectral Clustering']);
    dispIteration(' ', 1);
    disp('++++++ Server-side update ---> Parameter Average ++++++');
    disp('****** Update the matrix Z...');
    Z_new = zeros(m, d);
    for i = A
        Z_new = Z_new + Zs{i, 1};
    end
    Z_new = Z_new/length(A);
    disp('****** Evaluate the objective function......');
    
    Kzx=kernel(Z_new,X,sigma2);
    Kzz=kernel(Z_new,Z_new,sigma2);
    [obj, J_new] = dispLogInfo(s, obj, compute_obj, J, C, C_new, Z, Z_new, Kxx, Kzx, Kzz, lambda, prt);
    C = C_new;
    Z = Z_new;
    try
        J = J_new;
    catch
        disp('Terminated by too big noise.');
        break;
    end
    
    disp('****** Server-side parameter aggregation finished');
    % disp(['****** Select a set of clients A^s (with size |A^s| = ', num2str(sampling_m), ') based on the sample noise - ', Sampling_noise]);
    % A = samplingClients(sampling_m, PMF, 'nonrep');% sampling clients without replacement
    disp('****** Brodcast Z^s to all clients');
    
    dispIteration(' ', 1);
    
    %% client
    disp('++++++ Client-side update ---> Local optimization ++++++');
    for p = 1: nclients
        dispIteration(' ', 1);
        disp(['****** Client ', num2str(p), ': local update start...']);
        disp(['****** Client ', num2str(p), ' starts to receive the global matrix Z...']);
        Zs{p, 1} = Z;
        
        Kzx=kernel(Zs{p, 1},X(:, Xmask{p, 1}),sigma2);
        Kzz=kernel(Zs{p, 1},Zs{p, 1},sigma2);
        disp(['****** Client ', num2str(p), ' starts to update the local matrix C...']);
        C_new(:, Xmask{p, 1}) = localUpdateC(Kzx, Kzz, d, lambda);

        disp(['****** Client ', num2str(p), ' starts to update the local matrix Z...']);
        Zs{p, 1} = localUpdateZ(X(:, Xmask{p, 1}), Zs{p, 1}, Kzx, Kzz, C_new(:, Xmask{p, 1}), sigma2, eta);

        disp(['****** Denote Z_', num2str(p), '^', num2str(s), ' = Z_', num2str(p), '^{', num2str(s), ', Q^', num2str(s), '} and C_', num2str(p), '^', num2str(s), ' = C_', num2str(p), '^{', num2str(s), ', Q^', num2str(s), '}']);
        if ismember(p, A) == 1
            disp(['****** Upload Z_', num2str(p), '^', num2str(s), ' to the server']);
        end
    end
end

end

%%
function C_new = localUpdateC(Kzx, Kzz, d, lambda)

C_new= (Kzz+lambda*eye(d))\Kzx;

end

%%
function Z_new = localUpdateZ(X, Z, Kzx, Kzz, C_new, sigma2, eta)
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

%%
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

%%
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

%%
function Xr = splitClients(ndata, nclients)
% randomly partition n data points into n clients
% guarantee that each client has at least one data point

% each client has at least one data point
% method: plate insertion
shuffle = randperm(ndata);
pacesetter = sort(randperm(ndata, nclients - 1));
Xr = cell(nclients, 1);
for i = 1: nclients
    if i == 1
        Xr{i, 1} = shuffle(1: pacesetter(i));
    elseif i == nclients
        Xr{i, 1} = shuffle(pacesetter(i - 1) + 1: ndata);
    else
        Xr{i, 1} = shuffle(pacesetter(i - 1) + 1: pacesetter(i));
    end
end

end

%%
function [obj, J_new] = dispLogInfo(iter, obj, compute_obj, J, C, C_new, Z, Z_new, Kxx, Kzx, Kzz, lambda, prt)
%
formatSpec = '%.2e';
%
if size(Z, 1) <8000
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

%%
function dispIteration(MsgBox, prt)

if prt == 1
    disp(MsgBox);
end

end
