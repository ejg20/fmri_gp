%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Elad Gilboa
% 2013
%
% gpr_cov_grid_dn()
%
% Fast covariance matrix calculation for GP-grid
%
% Usage: [nlml, dnlml, alpha_kron] = gpr_cov_grid_dn(logtheta, xgrid, y, noise, cov)
%
% Inputs:
%       logtheta    hyperparameters vector
%       Xgrid       cell array of per dimension grid points
%       y           targets corresponding to inputs implied by xgrid
%       noise_var   noise variance of observations
%       cov         covariance function as in gpml-matlab
%
% Outputs:
%       nlml        negative log marginal likelihood
%       dnlml       nlml derivative wrt hyperparameters
%       alpha_kron  vector equivalence of the (K^-1)y to use for prediction
%       Qs
%       V_kron
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [nlml, dnlml, alpha_kron, Qs, V_kron] = gpr_cov_grid_dn_LRApprox(logtheta, xgrid, input, noise_var, cov)

%%%
% function parameters
%
% max allowed iteration for PCG
MAXITER = 2000;
%
%%%

% number of elements
N = prod(cellfun(@length, xgrid));
% number of real locations
n = length(input.index_to_N);
% data in real locations
y = input.data;
% number of dimensions
D = length(xgrid);
% ratio # observed to # of full grid elements
R = n/N;
% dummy noise variance coefficient
% dummyNoise = 1;
% check for valid observation vector
if size(y,1) ~= n
    error('Invalid vector of targets, quitting...');
end
% if length(logtheta) > D+2 || length(logtheta) < D+1
%     error('Error: Number of parameters do not agree with covariance function!')
% end

sphericalNoise = false;
learnNoiseFlag = false;

% if added a hyperparameter for noise sigma, then learn a single noise
% parameter
if(length(logtheta) == D+2)
    learnNoiseFlag = true;
    %check if the noise matrix is an eye matrix (spherical noise)
    if(prod(double(noise_var ==ones(length(noise_var),1))))
        sphericalNoise = true;
    end
    
end

% define objects for every dimension
Ks = cell(D,1);
Qs = cell(D,1);
QTs = cell(D,1);
Vs=cell(D,1);
V_kron = 1;
G=zeros(D+1,1);
% for calculation of derivatives later on
G(D+1)=1;
% decompose analysis for each dimension
for d = 1:D
    % use input locations from specific dimension
    xg = xgrid{d};
    % make sure its a column vector
    if size(xg,2) > size(xg,1)
        xg = xg';
    end
    % hyperparameters vector for this dimension. The signal variance is
    % assumed equal for all directions
    hyp.cov = [logtheta(d); logtheta(D+1)/D];
    % calculate covariance matrix using gpml. Since the covariance matrices
    % are for a signal dimension, they will be much smaller than the
    % original covariance matrix
    K_kron = feval(cov{:},hyp.cov, xg); %TODO: Toeplitz
    % save covariance matrix for later use
    Ks{d} = K_kron;
    % eigendecomposition of covariance matrix
    [Q,V] = eig(K_kron); %TODO: Toeplitz
    Qs{d} = Q;
    QTs{d} = Q';
    % make V a column vector
    V = (diag(V));
    Vs{d}=V;
    % the eigenvalues of the original covariance matrix are the kron
    % product of the single dimensions eigenvalues
    % this is a vector so still linear in memory
    V_kron = kron(V_kron, V);
    G(d) = length(xg);
end

[V_kron_sort V_index_to_N] = sort(real(V_kron),'descend');

if( sphericalNoise )
    gamma = exp(logtheta(D+2));
%     V_kron_noise = V_kron_sort + noise_var +1e-10;   %epsilon for computational stability
    %fprintf('Inverse condition number = %e\n', min(V_kron)/max(V_kron));
    alpha_kron = kron_mv(QTs, y(1:N));
    alpha_kron = alpha_kron./(V_kron + gamma);
    alpha_kron = kron_mv(Qs, alpha_kron);
    %alpha_k = conj_grad_solve(Qs, V_kron, noise_var*ones(N,1), y);
    
%     noise_var_approx = noise_var;
%     V_approx = V_kron_noise;
%     logdet_kron = sum(log(V_kron_noise));
    
else    % not spherical noise
    
    % if need to learn the noise then use the hyperparameter, noise is
    % approx [In 0; 0 inf*Iw]
    if(learnNoiseFlag)
        %         s2n = exp(2*logtheta(D+2));
        %         noise_var = s2n*noise_var;
        gamma = exp(logtheta(D+2));
%         noise_var = dummyNoise*sn*ones(size(noise_var));
        noise_var(input.index_to_N) = gamma;
%         gamma = real(exp(sum(log(noise_var))/N));
    else
        gamma = real(exp(sum(log(noise_var(input.index_to_N)))/n));
    end
    % add epsilon for computational stability
    %     V_kron = V_kron + 1e-10;
    %fprintf('Inverse condition number = %e\n', min(V_kron)/max(V_kron));
    
    % preconditioning matrix for PCG
    % if used zeros(size(noise_var)) then PCG has trouble converging
    %     C = 1e-15*ones(size(noise_var));
    %     C(~mask(:)) = noise_var(~mask(:)).^(-0.5);
    C = noise_var(input.index_to_N).^(-0.5);
    
    % maximum iteration should not exceed number of elements
    %     max_iter = min(N,MAXITER);
    max_iter = MAXITER;
    % fast calculation of (K+sn)^-1*y using PCG
    % make threshold lower for better approximation
    [alpha_kron rs] = pre_conj_grad_solve(Qs, V_kron, noise_var,...
        input.data,input.index_to_N, C,max_iter,1e-2);
    % fprintf('Conjugate gradient solver converged in %i iterations. logtheta =[%d,%d,%d]\n',length(rs),logtheta(1),logtheta(2),logtheta(3));
    
    
    % if number of PCG iteration exceeded max_iter then flag it invalid
    if(length(rs) == max_iter)
        nlml = inf;
        dnlml = inf*ones(size(logtheta));
        return;
    end
    
    % sanity check for PCG, this is the naive way to calculate alpha_kron
    %     K = kron(Ks{1},kron(Ks{2},Ks{3}));
    %     K1 = K(input.index_to_N,input.index_to_N);
    %     alpha_kron_true = (K1+diag(noise_var(input.index_to_N)))\y;     %%naive alpha calculation
    %     logdet_kron_true = sum(log(eig(K1+diag(noise_var(input.index_to_N))))) %% naive logdet calculation
    
    % estimation for log(det()) calculation. We used the geometrical mean
    % isotropic noise as our estimation of the diagonal noise. Other
    % estimations can be used although this one worked best for us
    
    %     %geometrical mean  (without dummy locations)
    % %         noise_var_approx = real(exp(sum(log(noise_var(input.index_to_N)))/n));
    %     %geometrical mean  (with dummy locations)
    %     noise_var_approx = real(exp(sum(log(noise_var))/N));
    
    % approximated logdet with geometrical noise approximation
    %          logdet_kron = sum(log(V_kron+noise_var_approx))
    % logdet_kron = sum(log(V_kron+exp(sum(log(noise_var(input.index_to_N))/N))));
    
    % approximated logdet using a low rank approximation for K
    %     tic
    %     [logdet_kron,Qm,Vm,Qz,Vz] = lowRankApproxForLogDet(Qs,V_kron,noise_var,input,0.85);
    %     toc
    %         logdet_kron
    
    
    %     phi = (cumsum(V_kron_sort)/sum(V_kron_sort));
    %     m = find(phi>0.99,1);
    %     phi_n = phi(n);
    %     V_approx = R/phi_n*V_kron_sort(1:n);
    %     %     logdet_kron =  m/N*real(sum(log(V_approx+noise_var_approx)));
    %     logdet_kron =  sum(log(V_approx+noise_var_approx))
    %      return
    
    %     % plot of results
%     G1 = size(Ks{1},1);
%     G2 = size(Ks{2},1);
%     G3 = size(Ks{3},1);
%     a = zeros(G2,G1);
%     a(input.index_to_N) = real(alpha_kron);
%     figure(10);imagesc(a(:,:,1)'); xlabel(num2str(logtheta'));colorbar;drawnow
%     figure(20);plot(phi)
    %         figure(11);imagesc((reshape(real(alpha_kron.*y),G2,G1)))
end

V_N = sum(V_kron_sort);
V_n = sum(V_kron_sort(1:n));
phi = V_n/V_N;
% gamma = real(exp(sum(log(noise_var(input.index_to_N)))/n));
Z = R/phi*V_kron_sort(1:n)+gamma;
logdet_kron = real(sum(log(Z)));
dlogdet_dZ=1./Z;       %n vector
dZ_dphi = -R*V_kron_sort(1:n)/phi^2; %n vector
dphi_dv = [ones(n,1);zeros(N-n,1)]*1/V_N - V_n/V_N^2; %N vector
dZ_dV = R/phi*ones(n,1); %n vector


% calculation of negative log marginal likelihood
% approximated using the logdet approximation
datafit = ((alpha_kron')*y);
complex_penalty = (logdet_kron);
nlml = 0.5*real( datafit+complex_penalty + n*log(2*pi) );
% if(complex_penalty<0)
%     stop;
% end

% approximation without the penalty, good for less smooth
% nlml = real(0.5*((alpha_kron')*y  + n*log(2*pi)));


% disp([num2str(alpha_kron'*y),' ',num2str(logdet_kron),' ',num2str(real(nlml))]);

% if nlml is inf then there is no need to calculate its derivatives
if(nlml == -inf)
    nlml = inf;
    dnlml = inf*ones(size(logtheta));
    return
end

%Now for the derivatives
dnlml = zeros(size(logtheta));

alpha_kron_to_N = zeros(N,1);
alpha_kron_to_N(input.index_to_N) = alpha_kron;

%lengthscales - l
% there are D lengthscale hyperparameter
for ell_outloop = 1:D
    % dK - derivative of covariance matrix
    dK = cell(D,1);
    dV_dtheta = 1;
    for ell_innerloop = 1:D
        % use input locations from specific dimension
        xg = xgrid{ell_innerloop};
        % make sure its a column vector
        if size(xg,2) > size(xg,1)
            xg = xg';
        end
        hyp.cov = [logtheta(ell_innerloop); logtheta(D+1)/D];
        % only calculate derivative for the ell hyperparameter of current
        % dimension. Otherwise use the covariance matrix from previous step
        if ell_innerloop == ell_outloop
            % calculate derivative using gpml, 1 = ell
            dK_kron = feval(cov{:},hyp.cov, xg,[],1);
        else
            dK_kron = Ks{ell_innerloop};
        end
        dK{ell_innerloop} = dK_kron;
        % transform dK using the eigenvectors calculated before
        dV = QTs{ell_innerloop}*dK_kron'*Qs{ell_innerloop};
        dV_dtheta = kron(dV_dtheta, diag(dV));
    end
    
    %     dnlml(ell_outloop) = 0.5*(sum(diag_Z./(V_kron+noise_var_approx)) - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
    %     dnlml(ell_outloop) = 0.5*(0 - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
    %     tic
    %     [tr_LRapprox] =  lowRankApproxForTrace(dK,Qm,Vm,Qz,Vz);
    %     dnlml(ell_outloop) = 0.5*(tr_LRapprox - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
    %     toc
    tr_approx_l= dlogdet_dZ'*(dZ_dphi*(dphi_dv'*dV_dtheta(V_index_to_N))+dZ_dV.*dV_dtheta(V_index_to_N(1:n)));
    dnlml(ell_outloop) = 0.5*(tr_approx_l - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
    %
    
end

% signal variance - sf
% dK - derivative of covariance matrix
dK = cell(D,1);
% dC - transform dK using the eigenvectors calculated before
% dC = cell(D,1);
dV_dtheta = 1;
for d = 1:D
    % use input locations from specific dimension
    xg = xgrid{d};
    % make sure its a column vector
    if size(xg,2) > size(xg,1)
        xg = xg';
    end
    hyp.cov = [logtheta(d); logtheta(D+1)/D];
    % calculate derivative using gpml, 2 = sf
    dK_kron = feval(cov{:},hyp.cov, xg,[],2);
    dK{d} = dK_kron;
    % transform dK using the eigenvectors calculated before
    dV = QTs{d}*dK_kron'*Qs{d};
    dV_dtheta = kron(dV_dtheta, real(diag(dV)));
end
% dnlml(D+1) = 0.5*(sum(diag_Z./(V_kron+noise_var_approx)) - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
% dnlml(D+1) = 0.5*(0 - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
% dnlml(D+1) = (sum(diag_Z./(V_kron+gavg_noise)) - alpha_kron'*kron_mv(dK, alpha_kron));
% tic
% [tr_LRapprox] =  lowRankApproxForTrace(dK,Qm,Vm,Qz,Vz);
% dnlml(D+1) = 0.5*(tr_LRapprox - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
% toc

% sanity check
% dK_full = kron(dK{1},kron(dK{2},dK{3}));
% dK1 = dK_full(input.index_to_N,input.index_to_N);
% tr_true = trace((K1+diag(noise_var(input.index_to_N)))\dK1)

% tr_approx_sf = R^(1+R)*sum(diag_Z(V_index_to_N(1:n))./(R^(1+R)*V_kron_sort(1:n)+noise_var_approx))
% tr_approx_sf = R^(1)*sum(diag_Z(V_index_to_N(1:n))./(R^(1)*V_kron_sort(1:n)+noise_var_approx))
% tr_approx_sf = R^(2)*sum(diag_Z(V_index_to_N(1:n))./(V_approx+noise_var_approx))
tr_approx_sf= dlogdet_dZ'*(dZ_dphi*(dphi_dv'*dV_dtheta(V_index_to_N))+dZ_dV.*dV_dtheta(V_index_to_N(1:n)));
dnlml(D+1) = 0.5*(tr_approx_sf - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));



% noise
if(learnNoiseFlag)
    %     sn = exp(logtheta(D+2));
    %     dnlml(D+2) = 0.5*(sum(1./(V_kron+noise_var_approx)) - alpha_kron'*alpha_kron);
    %     dnlml(D+2) = 0.5*(0 - alpha_kron'*alpha_kron);
    %     dnlml(D+2) = s2n*(sum(1./(V_kron+gavg_noise)) - alpha_kron'*alpha_kron);
    dnlml(D+2) = 0.5*(sum(dlogdet_dZ) - alpha_kron'*alpha_kron);
    
    %     tic
    %     [tr_LRapprox] =  lowRankApproxForTrace({},Qm,Vm,Qz,Vz);
    %     dnlml(D+2) = 0.5*(tr_LRapprox - alpha_kron'*alpha_kron);
    %     toc
    
end

% just a test need to take off
dnlml = 0.5*real(dnlml);
% dnlml = real(dnlml);
%  disp(logtheta');
% disp([nlml,datafit,complex_penalty,dnlml']);

%  stop = 1;


