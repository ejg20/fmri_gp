%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% gp_grid_prodcov()
%
% Fast product covariance matrix calculation for GP-grid. If grid is not complete
% or if noise is not homogeneus, use Nystrom approximation. If numeber of
% datapoints is very low, use PHI for better approximation.
%
% Usage: [nlml, dnlml, alpha_kron, Qs, V_kron] = gp_grid_prodcov(hypvec, input, gpmodel)
%
% Inputs:
%       hypvec      hyperparameters vector
%       input       gp_grid_input_class
%       gpmodel     gp_grid_gpmodel_class
%                   
%
% Outputs:
%       nlml        negative log marginal likelihood
%       dnlml       nlml derivative wrt hyperparameters
%       alpha_kron  vector equivalence of the (K^-1)y to use for prediction
%       Qs
%       V_kron
%
%
% SANITY CHECK - gradient of likelihood function,
% gp_grid_prodcov - PASSED
% checkgrad('gp_grid_prodcov_v2', hypers_init, 1e-3, input, gpmodel)
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [nlml, dnlml, alpha_kron, prodKs, nlml_skil] = gp_grid_prodcov_v2_hetero(hypvec, input, gpmodel, varargin)


mlock
persistent alpha_prev; % change to persistent!!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function parameters
%
% max allowed iteration for PCG
MAXITER = 1000;
% use phi (ratio of sums of eigenvalues) in approximation
USEPHI = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pnames = {'nlmlonly','Skiling'};
dflts =  {false, false};
[nlmlonly, Skiling]  = internal.stats.parseArgs(pnames, dflts, varargin{:});


% global alpha_prev;

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


% check for valid observation vector
if size(y,1) ~= n
    error('Invalid vector of targets, quitting...');
end

if length(input.index_to_N) ~= n
    error('Size unmatch between input indices (index_to_N) and number of inputs , quitting...');
end

sphericalNoise = false;
learnNoiseFlag = false;


% NEED TO CHAGE, GAMMA*I+Dn
% if needs to learn a single noise hyperparameter. Can learn a single noise
% hyperparameters for all input locations (aka spherical noise), or a
% single noise parameter to only the locations which are not dummy
% locations.
if( isa(gpmodel.noise_struct,'gp_grid_noise_class') )
    if(gpmodel.noise_struct.learn)
        learnNoiseFlag = true;
    end
    %check if the noise matrix is an eye matrix (spherical noise)
    if(gpmodel.noise_struct.sphericalNoise)
        sphericalNoise = true;
    end
else error('noise_struct should be a struct');
end



%% ============= PRECALCULATE SMALL MATRICES FOR KRONECKER =============

prodKs(Z) = struct('Ks',[],'Qs',[],'QTs',[],'Vs',[],'V_kron',[],'V_kron_sort',[],'V_index_to_N',[],...
    'R',[],'sV_N',[],'sV_n',[],'phi',[],'dVs',[],'dZ_dphi',[],'dphi_dv',[],'dZ_dV',[]); 

% define objects for every product kernel
for z = 1:Z
    % define objects for every dimension
    Ks = cell(P,1);
    Qs = cell(P,1);
    QTs = cell(P,1);
    Vs=cell(P,1);
    V_kron = 1;
    G=zeros(P+1,1);
    hlpvars=cell(Z,P);
    % for calculation of derivatives later on
    G(P+1)=1;
    % decompose analysis for each dimension
    for p = 1:P
        % use input locations from specific dimension
        xg = input.xgrid{p};
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
        Ks{p} = 1/2*(K+K');         % avoid numerical errors, force kernel matrix symmetry
        hlpvars{z,p} = hlpvar;
        if(sum(isnan( Ks{p}(:)))>0)     % check if valid kernel matrix
            nlml = inf;
            dnlml = inf*ones(size(hypvec));
            return;
        end
        % eigendecomposition of covariance matrix
        [Q,V] = eig(Ks{p}); 
        Qs{p} = Q;
        QTs{p} = Q';
        % make V a column vector
        V = diag(V);
        Vs{p}=V;
        % the eigenvalues of the original covariance matrix are the kron
        % product of the single dimensions eigenvalues
        % this is a vector so still linear in memory
        V_kron = kron(V_kron, V);
        G(p) = length(xg);
    end
    [V_kron_sort V_index_to_N] = sort(real(V_kron),'descend');
    
    % save for for each product kernel
    prodKs(z).Ks = Ks;
    prodKs(z).Qs = Qs;
    prodKs(z).QTs = QTs;
    prodKs(z).Vs=Vs;
    prodKs(z).V_kron = V_kron;
    prodKs(z).V_kron_sort = V_kron_sort;
    prodKs(z).V_index_to_N = V_index_to_N;
end

%% ================== CALCULATE DATA FIT ==================
numofgroups = length(gpmodel.noise_struct.groups);
if( sphericalNoise && Z==1 && numofgroups == 1) % group == 1 means only one noise parameter for all inputs
    % in case where the input is grid complete with spherical noise
    % and only have one product kernel, then we can perform all 
    % computations exactly
    gamma = exp(2*hypvec(end));
    alpha_kron = kron_mv(QTs, y(1:N));
    alpha_kron = alpha_kron./(V_kron + gamma + 1e-12); % jitter for conditioning
    alpha_kron = kron_mv(Qs, alpha_kron);
    noise_struct_var = gamma*gpmodel.noise_struct.var(:);
    
else    % not spherical noise, or not grid complete, or a sum of product kernels
    
    % if need to learn the noise then use the hyperparameter, noise is
    % approx [eye(n) 0; 0 inf*eye(w)]. We don't really use the imaginary noise
    % in the calculation.
    
    noise_struct_var = gpmodel.noise_struct.var(:);
    
    if(learnNoiseFlag)
        
        if(numofgroups == 1)
            % homogeneous noise for all inputs
            gamma = exp(2*hypvec(end)); % TODO: NEED TO CHAGE, GAMMA*I+Dn !!!
      
        elseif(numofgroups == n)
            % different noise parameter for each voxel at each
            % timepoint!!
            gamma = exp(2*hypvec(end-n+1:end));
            
        else
            % different noise parameter for each voxels at all
            % timepoints
            
            gamma = nan(N,1);
            for grpi = 1:numofgroups
                gamma(gpmodel.noise_struct.groups{grpi}) = exp(2*hypvec(end-numofgroups+grpi));
            end
            gamma = gamma(input.index_to_N);
        end
        noise_struct_var(input.index_to_N) = gamma;
        C=ones(n,1);
    else
        
        % THIS IS THE OLD WAY TO HANDLE ANISOTROPIC NOISE
%         % calculate the geommetrical mean isotropic noise of the know noise 
%         % as an  estimation of the known diagonal noise for log(det()) 
%         % calculations. Other estimations can be used although this one 
%         % worked best for us
%         gamma = real(exp(sum(log(gpmodel.noise_struct.var(input.index_to_N)))/n));

       
       
        
        % preconditioning matrix for PCG, only important for diagonal noise
        % when using a known noise model (not learned)
        C = noise_struct_var(input.index_to_N).^(-0.5);
    end
    
    % maximum iteration should not exceed number of elements
    %         max_iter = round(max(N/2,MAXITER));
    max_iter = MAXITER;
%     max_iter = max(MAXITER,sqrt(N));
 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% APPROXIMATE ALPHA %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % fast calculation of (K+sn)^-1*y using PCG
    % lower threshold for more accurate approximation

%     [alpha_kron numofitr, rhoratio] = pre_conj_grad_solve_v2(prodKs, noise_struct_var,...
%         input.zeromeandata, input.index_to_N, C(:), max_iter, (1e-3).^Z);

    if(~nlmlonly)       % if only checking nlml then its not part of the optimization sequence so start with initial guess 0
        prescision = 1e-4;
        totalitr = 0;
        while(totalitr < 30)
            [alpha_kron numofitr, rhoratio] =pre_conj_grad_solve_v2(prodKs, noise_struct_var,...
                input.zeromeandata,input.index_to_N, C(:), max_iter, prescision, alpha_prev);
            prescision = prescision*0.1;
            totalitr = totalitr+numofitr;
            alpha_prev = alpha_kron;
        end
    else
        rhoratio = inf;
    end
 
    if(rhoratio > 0.001 || isnan(rhoratio))        % if previous guess was not good then start again from 0.
        [alpha_kron numofitr, rhoratio] = pre_conj_grad_solve_v2(prodKs, noise_struct_var,...
            input.zeromeandata, input.index_to_N, C(:), 2*max_iter, 1e-3);
    end
    alpha_prev = alpha_kron;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % if number of PCG iteration exceeded max_iter then flag it invalid
    if(numofitr == max_iter && rhoratio > 0.1)
        nlml = inf;
        nlml_skil = inf;
        dnlml = 0*ones(size(hypvec));
%         figure(2); showSMkernel_v2(hypvec, {gpmodel.hyps_in_d{1}}, input.Fs);
%         figure(3); showSMkernel_v2(hypvec, {gpmodel.hyps_in_d{2}}, input.Fs);
%         figure(4); showSMkernel_v2(hypvec, gpmodel.hyps_in_d, input.Fs);
%         drawnow;
        return;
    end
    
   
    
    % can plot of results for testing
    %     G1 = size(Ks{1},1);
    %     G2 = size(Ks{2},1);
    %     G3 = size(Ks{3},1);
    %     a = zeros(G2,G1);
    %     a(input.index_to_N) = real(alpha_kron);
    %     figure(10);imagesc(a(:,:,1)'); xlabel(num2str(logtheta'));colorbar;drawnow
    %     figure(20);plot(phi)
    %         figure(11);imagesc((reshape(real(alpha_kron.*y),G2,G1)))
end

%% ================== CALCULATE NYSTROM LOGDET APPROX ==================

% WE USE THE SORTED ANISOTRIPIC NOISE FOR AN UPPER BOUND ON COMPLEXITY
[noise_ascend, noise_ix] = sort(noise_struct_var(input.index_to_N));

V_Nys = 0;
for z = 1:Z
    prodKs(z).sV_N = sum(prodKs(z).V_kron_sort);
    prodKs(z).sV_n = sum(prodKs(z).V_kron_sort(1:n));
    if USEPHI                 % using phi we are able to get better results
        % ratio of total explained variance the true matrix (n) and the
        % approximation using the first n eigenvalues of the N (Kronecker)
        % matrix
        prodKs(z).phi = prodKs(z).sV_n/prodKs(z).sV_N;
        
    else
        prodKs(z).phi = 1;
    end
    % ratio # observed to # of full grid elements
    prodKs(z).R = n/N;
    
    V_Nys = V_Nys + prodKs(z).R/prodKs(z).phi*prodKs(z).V_kron_sort(1:n);
    % since V_Nys is descending and the noise is ascending, the complexity
    % penalty will be an upper bound
   
end
Z_Nys = V_Nys+noise_ascend;
complex_penalty_Nys = real(sum(log(Z_Nys)));

% precalculate derivatives
dlogdet_dZ=1./Z_Nys;       %n vector
for z = 1:Z
    prodKs(z).dZ_dphi = USEPHI*(-prodKs(z).R*prodKs(z).V_kron_sort(1:n)/prodKs(z).phi^2); %n vector
    prodKs(z).dphi_dv = USEPHI*([ones(n,1);zeros(N-n,1)]*1/prodKs(z).sV_N - prodKs(z).sV_n/prodKs(z).sV_N^2); %N vector
    prodKs(z).dZ_dV = prodKs(z).R/prodKs(z).phi*ones(n,1); %n vector
end
    

%% ================== CALCULATE NLML ==================

% alpha_kron = alpha_kron_true;
datafit = ((alpha_kron')*y); 

% if only interested in nlml value then use Skiling method for more
% accurate approximation
nlml_skil = [];
if(Skiling)
    complex_penalty_buf=NaN(10,1);
    for i = 1:length(complex_penalty_buf)
        [complex_penaltyi, xx] = mcLogDet_mvkron(prodKs, noise_struct_var, 1, 100, 1, input.index_to_N);
        complex_penalty_buf(i) = complex_penaltyi;
    end
    complex_penalty_Skil = median(complex_penalty_buf);
    datafit = ((alpha_kron')*y);  
    nlml_skil = 0.5*real( datafit + complex_penalty_Skil + n*log(2*pi) );
end

complex_penalty =complex_penalty_Nys;
        
nlml = 0.5*real( datafit + complex_penalty + n*log(2*pi) );

% %%%%% TAKE AWAY %%%%%
if(nlml < 0)
    stop = 1;
end
% complex_hadamard_tight = hadamards_ineq(KKs, Dn, input.index_to_N);
% diagK = kron(diag(KKs{1}{1}),diag(KKs{1}{2}));
% complex_hadamard = sum(log(diagK(input.index_to_N)+Dn(input.index_to_N)));
% nlml_true = 0.5*real( (alpha_kron_true')*y + complex_penalty_true + n*log(2*pi) );
% global cbuff
% if(~exist('cbuff'))
%     cbuff = [];
% end
% % cbuff = [cbuff;complex_penalty_true,complex_penalty_Skil,complex_penalty_Nys,complex_hadamard_tight,complex_hadamard,nlml,nlml_true];
% cbuff = [cbuff;complex_penalty_true,complex_penalty_Nys,nlml,nlml_true];
% %%%%%%%%%%%%%%%%%%%%%


% if nlml is inf then there is no need to calculate its derivatives
if(nlml == -inf)
    nlml = inf;
    dnlml = zeros(size(hypvec));
    if( false &&  strcmp(gpmodel.cov{1}{1}{1},'covSM1D'))
        figure(2); showSMkernel_v2(hypvec, {gpmodel.hyps_in_d{1}}, input.Fs);
        figure(3); showSMkernel_v2(hypvec, {gpmodel.hyps_in_d{2}}, input.Fs);
        figure(4); showSMkernel_v2(hypvec, gpmodel.hyps_in_d, input.Fs);
        drawnow;
    end
    return
end

 % SANITY CHECK - for PCG, this is the naive way to calculate alpha_kron
    if(false)
        K=0;
        for z = 1:Z
            K = K+kron(prodKs(z).Ks{1},prodKs(z).Ks{2});
        end
        K = K(input.index_to_N,input.index_to_N);
        %%% logdet_kron_true = sum(log(eig(K+diag(noise_struct_var(input.index_to_N)))));
        L = chol(K+diag(noise_struct_var(input.index_to_N))); %% naive logdet calculation
        alpha_kron_true = (L\(L'\y));     %%naive alpha calculation
        complex_penalty_true = 2*sum(log(diag(L)));
        nlml = 0.5*real(  ((alpha_kron_true')*y) + complex_penalty_true + n*log(2*pi) );
    end

%% ================== NOW FOR THE DERIVATIVES ==================
dnlml = zeros(size(hypvec));

if(~nlmlonly)
        
    % zero pad alpha in dummy locations
    alpha_kron_to_N = zeros(N,1);
    alpha_kron_to_N(input.index_to_N) = alpha_kron;
    
    % precalculate dV for use later
    for z = 1:Z
        prodKs(z).dVs = cell(P,1);
        % for each dimension
        for p = 1:P
            prodKs(z).dVs{p} = diag(prodKs(z).QTs{p}*prodKs(z).Ks{p}*prodKs(z).Qs{p});
        end
    end
    
    % PROX SKILING METHOD (DOESNT WORK WELL)
    % % precalculate K\xx to be used later
    % [invKxx, ~] = pre_conj_grad_solve(Qs, V_kron, noise_struct_var,...
    %                 xx,input.index_to_N, C(:),max_iter,1e-2);
    % xxTxx=xx'*xx;        %O(N)
    
    for z = 1:Z
        for p = 1:P
            hyps_in_d = gpmodel.hyps_in_d{z}{p};
            hyp_val = hypvec(hyps_in_d);
            cov = gpmodel.cov{z}{p};
            % since the derivative will only effect this dimension, for all other
            % dimensions we can used our stored kernel matrices.
            dK = prodKs(z).Ks;
            %     dQs = Qs;
            %     dQTs = QTs;
            %     dVs = Vs;
            dV_Nys = prodKs(z).dVs;
            
            for hypd_i = 1:length(hyp_val)
                % use saved hlpvar for faster runtime.
                hlpvar=hlpvars{z,p};
                % use input locations from specific dimension
                xg = input.xgrid{p};
                % make sure its a column vector
                xg = xg(:);
                if(isempty(hlpvar))
                    dKtemp = feval(cov{:},hyp_val, xg,[],hypd_i);
                    dK{p} = (dKtemp+dKtemp')/2;
                else
                    dKtemp  = feval(cov{:},hyp_val, xg,[],hypd_i, hlpvar);
                    dK{p} = (dKtemp+dKtemp')/2;
                end
                %         [Q,V] = eig(dK{p});
                %         dQs{p} = Q;
                %         dQTs{p} = Q';
                %         V = diag(V);
                %         dVs{p}=V;
                % the eigenvalues of the original covariance matrix are the kron
                % product of the single dimensions eigenvalues
                % this is a vector so still linear in memory
                
                dV_Nys{p} = diag(prodKs(z).QTs{p}*dK{p}*prodKs(z).Qs{p});
                dV_dtheta = 1;
                %         dV_kron=1;
                for d_innerloop = 1:P
                    dV_dtheta = kron(dV_dtheta, dV_Nys{d_innerloop});
                    
                    %             dV_kron = kron(dV_kron, dVs{d_innerloop});
                end
                %         dV_kron_sqrt = sqrt(dV_kron);
                
                % SANITY CHECK - for derivative, this is the naive way to calculate the derivative
                %         K = kron(Ks{1},Ks{2}); K = K(input.index_to_N,input.index_to_N);
%                         dK1 = kron(dK{1},dK{2}); dK1 = dK1(input.index_to_N,input.index_to_N);
%                         tr_true = trace(L\(L'\dK1));
                
                tr_approx_Nys= dlogdet_dZ'*( ...
                    prodKs(z).dZ_dphi*(prodKs(z).dphi_dv'*dV_dtheta(V_index_to_N)) + ...
                    prodKs(z).dZ_dV.*dV_dtheta(V_index_to_N(1:n)) );
                
                % PROX SKILING METHOD (DOESNT WORK WELL)
                %             % this is a fast approximation but is not true because inv(K)dK is
                %             % not positive semidefinite.
                %             invKxx_N = zeros(N,1);
                %             invKxx_N(input.index_to_N) = invKxx;
                %             dKinvKxx = kron_mv(dK,invKxx_N);
                %             xxdKinvKxx=xx'*dKinvKxx(input.index_to_N);        %O(N)
                %             tr_approx_Skil = n*xxdKinvKxx/xxTxx;
                
                % Doesn't work because dK is not psd
                %         xxN = randn(N,1);
                %         dVxxN =dV_kron_sqrt.*xxN;
                %         dQTdVxxN = kron_mv(dQTs,dVxxN);
                %
                %         [invKdQTdVxxN, ~] = pre_conj_grad_solve(Qs, V_kron, noise_struct_var,...
                %                 dQTdVxxN(input.index_to_N),input.index_to_N, C(:),max_iter,1e-2);
                %
                %         invKdQTdVxxN_toN = zeros(N,1);
                %         invKdQTdVxxN_toN(input.index_to_N)=invKdQTdVxxN;
                %
                %         dQinvKdQTdVxxN = kron_mv(dQs,invKdQTdVxxN_toN);
                %
                %         tr_approx2 = N*xxN'*(dV_kron_sqrt.*dQinvKdQTdVxxN)/(xxN'*xxN);
                
                %         dKKs{1} = dK;
                %         [dlogdet, ~] = mcLogDet_derivative_mvkron(KKs, Dn, dKKs, xx, 100, 1.1, input.index_to_N);
                
                tr_approx = tr_approx_Nys;
                
                dnlml(hyps_in_d(hypd_i)) = dnlml(hyps_in_d(hypd_i))+ 0.5*(tr_approx - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
                
                %             run('mcLogDet_test_mvkron')
            end
        end
    end
    
     
    % noise
    if(learnNoiseFlag)
        % Adjust for calculating derivatives in log space
        if(numofgroups == 1)
            % If homogeneous then only need to calculate a simple scalar
            % derivative
            dnlml(end) = (sum(dlogdet_dZ) - alpha_kron'*alpha_kron)*gamma;
        else
            % If learning heteroschedastic noise need to realign the sorted
            % dlogdet_dZ to match the original order of the noise.
            dlogdet_dZ_resort = dlogdet_dZ(noise_ix);
            
            dnoisevec = (dlogdet_dZ_resort - alpha_kron.^2)...
                    .*noise_struct_var(input.index_to_N);
            
            if(numofgroups == n)
                % different noise parameter for each voxel at each
                % timepoint!!
                dnlml(end-numofgroups+1:end) = dnoisevec;
                
            else
                % learning heteroscedastic noise for each voxel
                dnoisevec_to_N = zeros(N,1);
                dnoisevec_to_N(input.index_to_N)= dnoisevec;
                for grpi = 1:numofgroups
                    grpix = gpmodel.noise_struct.groups{grpi};
                    dnlml(end-numofgroups+grpi) = sum(dnoisevec_to_N(grpix));
                end
           
            end

        end
    end
    

    % just a test need to take off
    dnlml = real(dnlml);
end

if( false && strcmp(gpmodel.cov{1}{1}{1},'covSM1D'))
    % r = mvkronFrobenius(prodKs,input.index_to_N)
    % r = mvkronFrobenius(prodKs)
    % figure(2); showSMkernel_v2(hypvec, {gpmodel.hyps_in_d{1}}, input.Fs);
    figure(3); showSMkernel_v2(hypvec, {gpmodel.hyps_in_d{2}}, input.Fs);
    % figure(4); showSMkernel_v2(hypvec, gpmodel.hyps_in_d, input.Fs);
    drawnow;
end


end
