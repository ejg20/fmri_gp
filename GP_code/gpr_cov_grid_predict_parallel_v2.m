%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Elad Gilboa
% 2013
%
% gpr_cov_grid_predict_parallel()
%
% Fast perallel prediction for GP-grid by using alpha which is the vector
% equivalence of (K^-1)y calculated by gpr_cov_grid_dn
%
% Usage: [mu_f, std_f] = gpr_cov_grid_predict_multi(xstars, logtheta, xgrid, ygrid, alpha, cov)
%        [mu_f] = gpr_cov_grid_predict_multi(xstars, logtheta, xgrid, ygrid, alpha, cov)
%
% Inputs:
%       xstars      locations for prediction
%       logtheta    hyperparameters for covariance function
%       xgrid       cell array of per dimension grid points
%       ygrid       targets corresponding to inputs implied by xgrid
%       alpha       vector equivalence of (K^-1)y
%       cov         covariance function as in gpml-matlab
%
% Outputs:
%       mu_f        posterior mean for xstars locations
%       var_f       posterior variance for xstars locations 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mu_f, var_f] = gpr_cov_grid_predict_parallel_v2(xstars, hypvec, input, gpmodel, alpha, prodKs)

% tic
%%%
% function parameters
%
% max allowed iteration for PCG
MAXITER = 10000;
%
%%%

% if number of out arguments is 1 then only calculate posterior mean (much
% faster)
if(nargout > 1)
    std_flag = true;
else
    std_flag = false;
end

% number of elements
N = input.get_N();
% number of real locations
%n = length(input.index_to_N);
% data in real locations
%y = input.data(:);
% number of dimensions
P = input.get_P();
% check for valid observation vector
assert(P > 0);
assert(size(xstars.subs,2) == P);
% assert(size(ygrid,1) == N);
% number of prediction locations
M = size(xstars.subs,1);

mu_f = zeros(M,1);

% DummyLocations = logical(abs(alpha) < 0.01);

% since prediction calculations are usually very repetative we precalculate
% Kxy covariance matrices using only unique values for each dimension in
% xstarts
Kxy_sparse = cell(P,2);
Kxtxt = 1;
z=1
for p=1:P
    xg = input.xgrid{p};
    % make sure its a column vector
    if size(xg,2) > size(xg,1)
        xg = xg';
    end
    [b, ~, n] = unique(xstars.subs(:,p));
%     hyp_cov = [logtheta(d); logtheta(D+1)/D];
    hyps_in_d = gpmodel.hyps_in_d{z}{p};
    hyp_val = hypvec(hyps_in_d);
    
    Kxy_sparse{p,1} = feval(gpmodel.cov{z}{p}{:},hyp_val, xg, b);
    % save the reference to all xstars.subs with this input in this dimension
    Kxy_sparse{p,2} = n;
    if(std_flag)
        Kxtxt = Kxtxt*(feval(gpmodel.cov{:},hyp_val, 1));       % calculate variance of a single location
    end
end

% toc

var_f = zeros(M,1);
% preconditioning matrix for PCG
C = gpmodel.noise_struct.var(input.index_to_N).^(-0.5);
% maximum iteration should not exceed number of elements
max_iter = min(N,MAXITER);

index_to_N = input.index_to_N;
% tic
parfor m = 1:M %change to parfor
    
    Kstar_kron = 1;
    for p = 1:P
        % rebuild full matrix from sparse representation
        Kxy_d = Kxy_sparse{p,1}(:,Kxy_sparse{p,2}(m));
        % Kstar_kron = Kxx*
        Kstar_kron = kron(Kstar_kron, Kxy_d);
    end
%     toc
%     tic
    % calculate GP posterior mean Kxy*alpha = Kxy*(K+sn)^-1*y
    alpha_to_N = zeros(N,1);
    alpha_to_N(index_to_N)=alpha;
    mu_f(m) = real(Kstar_kron'*alpha_to_N);
    
%     if(std_flag)
%         % make sure Kstar_kron a column vector
%         if size(Kstar_kron,2) > size(Kstar_kron,1)
%             Kstar_kron = Kstar_kron';
%         end
%         % fast calculation of alpha_kron = (Kxx+sn^2)^-1*Kxx* using PCG
%         %[alpha_kron rs] = pre_conj_grad_solve(Qs, V_kron, noise, Kstar_kron_input, C, max_iter);
%         [alpha_kron rs] = pre_conj_grad_solve_wrapper(Qs, V_kron, noise, Kstar_kron, C, max_iter,index_to_N);
%         alpha_to_N = zeros(N,1);
%         alpha_to_N(index_to_N)=alpha_kron;
%         % calculate posterior variance = Kx*x* - Kx*x (Kxx-sn^2)^-1 Kxx*
%         var_f(m) = Kxtxt - Kstar_kron'*alpha_to_N;
%         
%     end
    
%     disp(['m=',num2str(m),' ',num2str(exctime)])
end
% m_exctime = toc
if(std_flag)
    
    tic
    
    noise_struct_var = gpmodel.noise_struct.var;
    % if the noise model hyperparamter was learned, then
    % get the new noise matrix
    if(gpmodel.noise_struct.learn == true)
            gamma2 = exp(2*hypvec(end));
            noise_struct_var = gpmodel.noise_struct.var*gamma2;
    end
    
    parfor m = 1:M     %change to parfor
        %     exctime_start = toc;
        Kstar_kron = 1;
        for p = 1:P
            % rebuild full matrix from sparse representation
            Kxy_d = Kxy_sparse{p,1}(:,Kxy_sparse{p,2}(m));
            % Kstar_kron = Kxx*
            Kstar_kron = kron(Kstar_kron, Kxy_d);
        end
        %     toc
        %     tic
        % calculate GP posterior mean Kxy*alpha = Kxy*(K+sn)^-1*y
        %     alpha_to_N = zeros(N,1);
        %     alpha_to_N(index_to_N)=alpha;
        %     mu_f(m) = real(Kstar_kron'*alpha_to_N);
        
        if(std_flag)
            % make sure Kstar_kron a column vector
            if size(Kstar_kron,2) > size(Kstar_kron,1)
                Kstar_kron = Kstar_kron';
            end
            % fast calculation of alpha_kron = (Kxx+sn^2)^-1*Kxx* using PCG
            %[alpha_kron rs] = pre_conj_grad_solve(Qs, V_kron, noise, Kstar_kron_input, C, max_iter);
            [alpha_kron, ~] = pre_conj_grad_solve(prodKs, noise_struct_var, Kstar_kron(index_to_N),index_to_N, C, max_iter, 1e-2);
            
            alpha_to_N = zeros(N,1);
            alpha_to_N(index_to_N)=alpha_kron;
            % calculate posterior variance = Kx*x* - Kx*x (Kxx-sn^2)^-1 Kxx*
            var_f(m) = Kxtxt - Kstar_kron'*alpha_to_N;
        end
        %     exctime = toc
        %     disp(['m=',num2str(m),' ',num2str(exctime-exctime_start)])
%                 disp(['m=',num2str(m)]);
        
    end
    v_exctime = toc
end

% save('predict_times','m_exctime','v_exctime');


