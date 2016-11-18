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

function [nlml, dnlml, alpha_kron, Qs, V_kron] = gpr_cov_grid_dn(logtheta, xgrid, input, noise_var, cov)

%%%
% function parameters
%
% max allowed iteration for PCG
MAXITER = 10000;
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
% check for valid observation vector
if size(y,1) ~= n
    error('Invalid vector of targets, quitting...');
end
if length(logtheta) > D+2 || length(logtheta) < D+1
    error('Error: Number of parameters do not agree with covariance function!')
end

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
    Vs{d}=V;
    % make V a column vector
    V = diag(V);
    % the eigenvalues of the original covariance matrix are the kron
    % product of the single dimensions eigenvalues
    % this is a vector so still linear in memory
    V_kron = kron(V_kron, V);
    
end

if( sphericalNoise )
    noise_var = exp(logtheta(D+2));
    V_kron_noise = V_kron + noise_var +1e-10;   %epsilon for computational stability
    %fprintf('Inverse condition number = %e\n', min(V_kron)/max(V_kron));
    alpha_kron = kron_mv(QTs, y(1:N));
    alpha_kron = alpha_kron./V_kron_noise;
    alpha_kron = kron_mv(Qs, alpha_kron);
    %alpha_k = conj_grad_solve(Qs, V_kron, noise_var*ones(N,1), y);
    
    logdet_kron = sum(log(V_kron_noise));
    
    noise_var_approx = noise_var;
    
else    % not spherical noise
    
    % if need to learn the noise then use the hyperparameter, noise is
    % approx [In 0; 0 inf*Iw]
    if(learnNoiseFlag)
        %         s2n = exp(2*logtheta(D+2));
        %         noise_var = s2n*noise_var;
        sn = exp(logtheta(D+2));
        noise_var(input.index_to_N) = sn;
    end
    % add epsilon for computational stability
    V_kron = V_kron + 1e-10;
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
    [alpha_kron rs] = pre_conj_grad_solve(Qs, V_kron, noise_var, input, C,max_iter);
    % fprintf('Conjugate gradient solver converged in %i iterations. logtheta =[%d,%d,%d]\n',length(rs),logtheta(1),logtheta(2),logtheta(3));
    
    
    % if number of PCG iteration exceeded max_iter then flag it invalid
    if(length(rs) == max_iter)
        nlml = inf;
        dnlml = inf*ones(size(logtheta));
        return;
    end
    
    % sanity check for PCG, this is the naive way to calculate alpha_kron
    % alpha_kron = (kron(Ks{1},Ks{2})+diag(noise))\y;     %%naive alpha calculation
    
    % estimation for log(det()) calculation. We used the geometrical mean
    % isotropic noise as our estimation of the diagonal noise. Other
    % estimations can be used although this one worked best for us
    
    %geometrical mean  (without dummy locations)
    %     noise_var_approx = exp(sum(log(noise_var(input.index_to_N)))/n);
    %geometrical mean  (with dummy locations)
    noise_var_approx = exp(sum(log(noise_var))/N);
    
    logdet_kron = sum(log(V_kron+noise_var_approx));
    
    % sanity check, naive logdet calculation
    % logdet_kron = sum(log( eig(kron(Ks{1},Ks{2})+diag(noise))));
    
    % plot of results
    G1 = size(Ks{1},1);
    G2 = size(Ks{2},1);
    G3 = size(Ks{3},1);
    a = zeros(G3,G2,G1);
    a(input.index_to_N) = real(alpha_kron);
    figure(10);imagesc(a(:,:,1)'); xlabel(num2str(logtheta'));drawnow
    %     figure(11);imagesc((reshape(real(alpha_kron.*y),G2,G1)))
end

% calculation of negative log marginal likelihood
nlml = real(0.5*((alpha_kron')*y + (logdet_kron) + n*log(2*pi)));

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
    diag_Z = 1;
    for ell_innerloop = 1:D
        % use input locations from specific dimension
        xg = xgrid{ell_innerloop};
        % make sure its a column vector
        if size(xg,2) > size(xg,1)
            xg = xg';
        end
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
        dC = QTs{ell_innerloop}*dK_kron*Qs{ell_innerloop};
        diag_Z = kron(diag_Z, diag(dC));
    end
    
    diag_Z_kron_to_N = zeros(N,1);
    diag_Z_kron_to_N(input.index_to_N) = diag_Z(input.index_to_N);
    dnlml(ell_outloop) = 0.5*(sum(diag_Z_kron_to_N./(V_kron+noise_var_approx)) - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
    %     dnlml(ell_outloop) = 0.5*(sum(diag_Z./(V_kron+noise_var_approx)) - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
end

% signal variance - sf
% dK - derivative of covariance matrix
dK = cell(D,1);
% dC - transform dK using the eigenvectors calculated before
% dC = cell(D,1);
diag_Z = 1;
diag_K=1;
for d = 1:D
    % use input locations from specific dimension
    xg = xgrid{d};
    % make sure its a column vector
    if size(xg,2) > size(xg,1)
        xg = xg';
    end
    % calculate derivative using gpml, 2 = s2f
    dK_kron = feval(cov{:},hyp.cov, xg,[],2);
    dK{d} = dK_kron;
    % transform dK using the eigenvectors calculated before
    dC = QTs{d}*dK_kron*Qs{d};
    diag_Z = kron(diag_Z, diag(dC));
    
    diag_K = kron(diag_K, diag(dK_kron));
end
diag_Z_kron_to_N = zeros(N,1);
diag_Z_kron_to_N(input.index_to_N) = diag_Z(input.index_to_N);
dnlml(D+1) = 0.5*(sum(diag_Z_kron_to_N./(V_kron+noise_var_approx)) - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
%dnlml(D+1) = 0.5*(sum(diag_Z./(V_kron+noise_var_approx)) - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
% dnlml(D+1) = (sum(diag_Z./(V_kron+gavg_noise)) - alpha_kron'*kron_mv(dK, alpha_kron));

% % input_sf.data = diag_K(input.index_to_N);
% % input_sf.index_to_N = input.index_to_N;
% % [alpha_kron_sf rs] = pre_conj_grad_solve(Qs, V_kron, noise_var, input_sf, C,max_iter);
% % dnlml_sf = 0.5*(sum(alpha_kron_sf) - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));


% noise
if(learnNoiseFlag)
    diag_Z_kron_to_N = zeros(N,1);
    diag_Z_kron_to_N(input.index_to_N) = 1;
    %     sn = exp(logtheta(D+2));
    dnlml(D+2) = 0.5*(sum(diag_Z_kron_to_N./(V_kron+noise_var_approx)) - alpha_kron'*alpha_kron);
%     dnlml(D+2) = 0.5*(sum(1./(V_kron+noise_var_approx)) - alpha_kron'*alpha_kron);
    %     dnlml(D+2) = s2n*(sum(1./(V_kron+gavg_noise)) - alpha_kron'*alpha_kron);
end

% just a test need to take off
% dnlml = 0.1*real(dnlml);
%  disp(logtheta');
 disp([nlml,dnlml']);

%  stop = 1;


