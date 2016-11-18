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
function [mu_f, var_f] = gpgrid_predict_fullkron(xstarsinput, hypvec, input, gpmodel, alpha, prodKs)

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
n = input.get_n();
% data in real locations
y = input.zeromeandata(:);
% number of dimensions
P = input.get_P();
% number of summed product kernels
Z = length(gpmodel.hyps_in_d);

% dummy noise variance coefficient
% dummyNoise = 1;
% check for valid observation vector
if size(y,1) ~= n
    error('Invalid vector of targets, quitting...');
end




%% ============= PRECALCULATE SMALL MATRICES FOR KRONECKER =============

% define objects for every product kernel
for z = 1:Z
    % define objects for every dimension
    Kstars = cell(P,1);
  
   
    % decompose analysis for each dimension
    for p = 1:P
        % use input locations from specific dimension
        xg = xstarsinput.xgrid{p};
        % make sure its a column vector
        xg = xg(:);
        
        % build hyperparameters vector for this dimension
        hyp_val = hypvec(gpmodel.hyps_in_d{z}{p});
        
        % the covariance function of the z's prodkernel, at p dimension
        cov = gpmodel.cov{z}{p};
        
        % calculate covariance matrix using gpml. Since the covariance matrices
        % are for a signal dimension, they will be much smaller than the
        % original covariance matrix
        hlpvar=[];
        if(nargout(cov{1}) == 1)
            [K] = feval(cov{:},hyp_val, xg); %TODO: CHANGE TO DIFFERENT COV FUNCTIONS!!
        elseif(nargout(cov{1}) == 2)
            [K, hlpvar] = feval(cov{:},hyp_val, xg); 
        end
        % save covariance matrix for later use
        Kstars{p} = 1/2*(K+K');         % avoid numerical errors, force kernel matrix symmetry
       
    end
    

end



var_f = zeros(M,1);
% preconditioning matrix for PCG
C = gpmodel.noise_struct.var(input.index_to_N).^(-0.5);
% maximum iteration should not exceed number of elements
max_iter = min(N,MAXITER);

index_to_N = xstartinput.index_to_N;
% tic

    alpha_to_N = zeros(N,1);
    alpha_to_N(index_to_N)=alpha;
    mu_N = real(Kstars,alpha_to_N);
    mu_f = mu_N(index_to_N);
    
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
% if(std_flag)
%     
%     tic
%     noise_struct_var = gpmodel.noise_struct.var
%     parfor m = 1:M     %change to parfor
%         %     exctime_start = toc;
%         Kstar_kron = 1;
%         for p = 1:P
%             % rebuild full matrix from sparse representation
%             Kxy_d = Kxy_sparse{p,1}(:,Kxy_sparse{p,2}(m));
%             % Kstar_kron = Kxx*
%             Kstar_kron = kron_mv(Kstar_kron, Kxy_d);
%         end
%         %     toc
%         %     tic
%         % calculate GP posterior mean Kxy*alpha = Kxy*(K+sn)^-1*y
%         %     alpha_to_N = zeros(N,1);
%         %     alpha_to_N(index_to_N)=alpha;
%         %     mu_f(m) = real(Kstar_kron'*alpha_to_N);
%         
%         if(std_flag)
%             % make sure Kstar_kron a column vector
%             if size(Kstar_kron,2) > size(Kstar_kron,1)
%                 Kstar_kron = Kstar_kron';
%             end
%             % fast calculation of alpha_kron = (Kxx+sn^2)^-1*Kxx* using PCG
%             %[alpha_kron rs] = pre_conj_grad_solve(Qs, V_kron, noise, Kstar_kron_input, C, max_iter);
%             [alpha_kron, ~] = pre_conj_grad_solve(prodKs, noise_struct_var, Kstar_kron(index_to_N),index_to_N, C, max_iter, 1e-2);
%             
%             alpha_to_N = zeros(N,1);
%             alpha_to_N(index_to_N)=alpha_kron;
%             % calculate posterior variance = Kx*x* - Kx*x (Kxx-sn^2)^-1 Kxx*
%             var_f(m) = Kxtxt - Kstar_kron'*alpha_to_N;
%         end
%         %     exctime = toc
%         %     disp(['m=',num2str(m),' ',num2str(exctime-exctime_start)])
% %                 disp(['m=',num2str(m)]);
%         
%     end
%     v_exctime = toc
% end

% save('predict_times','m_exctime','v_exctime');


