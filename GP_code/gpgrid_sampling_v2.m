%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% gpgrid()
%
% Fast version of GP prediction ALL points lie on a grid (with G grid points per dim).
% This version also attempts to do everything in linear memory complexity.
% gpgrid_img_dn does not consider spherical noise as a hyperparameter to be learned,
% but as input dependent noise specified by the user.
%
%
% Usage: [hypers_learned, nlml, Iout, Vout] = gpgrid(input, gpmodel, xstar, itrnum)
%        [hypers_learned, nlml] = gpgrid(input, gpmodel, xstar, itrnum)
%
% Inputs:
%       input       gp_grid_input_class
%       gpmodel     gp_grid_gpmodel_class
%       xstar       subscripts of target locations for prediction
%       itrnum      max iteration
%
% Outputs:
%     hypers_learned  learned hyperparameters
%     Iout              interpolated Image
%     Vout              GP variance (not implemented)
%
%
% Note:
% To use the package you must first install gpml-matlab.
% The package can be found in www.gaussianprocess.org/gpml/. The last
% version tested was gpml-matlab-v3.2-2013-01-15
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hypers_learned, results, Iout, Vout] = gpgrid_v2(input, gpmodel, xstar, itrnum, varargin)

if(nargin < 4)
    itrnum = 1000;
end

pnames = {'checkinit'};
dflts =  {false};
[checkinit]  = internal.stats.parseArgs(pnames, dflts, varargin{:});

% % allocate memory
% Iout = zeros(IoutSize(:)');
% If only one output than only return learned hyperparameters
% Otherwise, also perform prediction for Iout
if(nargout > 2)
    predictionFlag = true;
    if(nargout > 3)
        voutFlag = true;
    else
        voutFlag = false;
    end
else
    predictionFlag = false;
end
% if(~isempty(gpmodel.logpriorfuns)) % if no logfunc specified than use standard likelihood function
%     gpmodel.logpriorfuns = []; %@(t) gp_grid_prodcov( t, input.xgrid, input, noise_struct, cov, hyps_in_dim);
% end


results = [];

% prepare a results table with a row for every run, first column will be the
% nlml value and the rest will be the values of the optimal hyperparameters.
allResultTable = zeros(size(gpmodel.hyperparams,1),size(gpmodel.hyperparams,2)+2);

% loglikefunc = @(t) gp_grid_prodcov( t, input, gpmodel);
% loglikefunc = @(t) gp_grid_prodcov_newapprox( t, input, gpmodel);

%%%  TAKE OFF
% gpmodeltemp = gp_grid_gpmodel_class(gpmodel);
% gpmodeltemp.hyps_in_d =  {gpmodel.hyps_in_d};
%%%

loglikefunc = @(t) gp_grid_prodcov_v2( t, input, gpmodel);
logpostfunc = @(t) gp_grid_posterior(t,loglikefunc,gpmodel.logpriorfuns);

if(checkinit)
    hypers_init = gpmodel.hyperparams(1,:)';
    loglikefunc = @(t) gp_grid_prodcov_v2( t, input, gpmodel, 'nlmlonly', true, 'Skiling', true);
    [nlml_lik, ~, ~, ~, nlml_lik_skil] = loglikefunc( hypers_init );
    
    %     loglikefunc = @(t) gp_grid_prodcov_v2( t, input, gpmodel, 'nlmlonly', true, 'Skiling', false);
    %     logpostfunc = @(t) gp_grid_posterior(t,loglikefunc,gpmodel.logpriorfuns);
    %     nlml_pos = logpostfunc(hypers_init);
    
    results.nlml_lik = nlml_lik;
    results.nlml_lik_skil = nlml_lik_skil;
    %     results.nlml_pos = nlml_pos;
    
    hypers_learned = hypers_init;
    
    return;
    
elseif(isempty(gpmodel.learn) || gpmodel.learn==true)
    %     parfor run_i = 1:size(InitParamSet.vals,1)
    for run_i = 1:size(gpmodel.hyperparams,1)
        hypers_init = gpmodel.hyperparams(run_i,:)';
        
        n_hmc_samples = 400;
        hmc_options.num_iters = 1;
        hmc_options.Tau = 5;  % Number of steps.
        hmc_options.epsilon = 0.001;
        hmc_samples = NaN(length(hypers_init),n_hmc_samples);
        hmc_samples(:,1) = hypers_init(:);
        tic
        for n = 2:n_hmc_samples  
            % SANITY CHECK - gradient of cov function,
            % gp_grid_posterior - PASSED
            % checkgrad('gp_grid_posterior', hmc_samples(n - 1,:)', 1e-3,loglikefunc,priorfuns)

            [hmc_samples(:,n), nll, arate, tail{n}] = hmc( logpostfunc, hmc_samples(:,n - 1), hmc_options );
            n
        end
        results.runtime = toc;
        hypers_learned = hmc_samples(:,n);
        allResultTable(run_i,:) = [nll,n_hmc_samples,hypers_learned'];
        save('gp_grid_temp_hypers','allResultTable');
    end
    sortResultTable = sortrows(allResultTable,1);
    hypers_learned = sortResultTable(1,3:end)';
    results.nlml = sortResultTable(1,1);
    results.numofitr = sortResultTable(1,2);
else
    hypers_learned = gpmodel.hyperparams(1,:)';
    results.nlml = logpostfunc(hypers_learned);
end

% disp(['sf learned = 2^',num2str(imin-1),' ells learned = 2^',num2str(imin_learnedtt(imin)-1)]);
% disp((hypers_learned)./log(2))
% disp(exp(hypers_learned))

% pause
if(predictionFlag)
    
    [nlml, ~, alpha_kron, prodKs] = loglikefunc( hypers_learned );
    
    %learnexectime =toc
    disp(nlml);
    % if the noise model was spherical and the hyperparamter was learned, then
    % get the new noise matrix
    if(gpmodel.noise_struct.learn == true)
        gamma2 = exp(2*hypers_learned(end));
        gpmodel.noise_struct.var = gpmodel.noise_struct.var*gamma2;
    end
    
    
    % use meshgid to create a locations for interpolation
    % we interpolate also observed locations for denoising
    
    % allocate memory for variance of prediction
    %     Iout = zeros(size(IoutSize));
    %     Vout = zeros(size(IoutSize));
    
    % perform prediction using covMatern covariance function
    if(voutFlag)
        [mu_f, var_f] = gpr_cov_grid_predict_parallel_v2(xstar, hypers_learned, input, gpmodel, alpha_kron, prodKs);
    else
        [mu_f] = gpr_cov_grid_predict_parallel_v2(xstar, hypers_learned, input, gpmodel, alpha_kron, prodKs);
        var_f = 0;
    end
    Iout = mu_f;
    Vout = var_f;
    
end



end