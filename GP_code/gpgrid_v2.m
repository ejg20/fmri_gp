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

pnames = {'checkinit','params'};
dflts =  {false,[]};
[checkinit, params]  = internal.stats.parseArgs(pnames, dflts, varargin{:});

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

% global prodcovfunc predfunc

if(exist('params') && isfield(params,'prodcovfunc') && ~isempty(params.prodcovfunc))
    prodcovfunc = params.prodcovfunc;
else
    prodcovfunc = 'gp_grid_prodcov_v2';   
end

if(exist('params') && isfield(params,'predfunc') && ~isempty(params.predfunc))
    predfunc = params.predfunc;
else
    predfunc = 'gpr_cov_grid_predict_parallel_v2';   
end

loglikefunc = @(t) feval(prodcovfunc, t, input, gpmodel);
logpostfunc = @(t) gp_grid_posterior(t,loglikefunc,gpmodel.logpriorfuns);


if(checkinit)
    hypers_init = gpmodel.hyperparams(1,:)';
    loglikefunc = @(t) feval(prodcovfunc, t, input, gpmodel, 'nlmlonly', true, 'Skiling', false);
    [nlml_lik, ~, ~, ~, nlml_lik_skil] = loglikefunc( hypers_init );
    
%     loglikefunc = @(t) feval(prodcovfunc, t, input, gpmodel, 'nlmlonly', true, 'Skiling', false);
%     logpostfunc = @(t) gp_grid_posterior(t,loglikefunc,gpmodel.logpriorfuns);
%     nlml_pos = logpostfunc(hypers_init);
    
    results.nlml_lik = nlml_lik;
    results.nlml_lik_skil = [];
    results.nlml = nlml_lik;
%     results.nlml_pos = nlml_pos;
    
    hypers_learned = hypers_init;
    
else
    
    if(isempty(gpmodel.learn) || gpmodel.learn==true)
        %     parfor run_i = 1:size(InitParamSet.vals,1)
        for run_i = 1:size(gpmodel.hyperparams,1)
            hypers_init = gpmodel.hyperparams(run_i,:)';

            p.length =    -itrnum;
            p.method =    'BFGS';% 'BFGS' 'LBFGS' or 'CG'
            p.SIG = 0.1;
            p.verbosity = 1; %0 quiet, 1 line, 2 line + warnings (default), 3 graphical
%             p.mem        % number of directions used in LBFGS (default 100)
            tic
            [hypers_learned fx, numofitr] = minimize_new(hypers_init, logpostfunc, p );
            results.runtime = toc;
            %         disp(hypers_learned);
            allResultTable(run_i,:) = [fx(end),numofitr,hypers_learned'];
            save('gp_grid_temp_hypers','allResultTable');
        end
        sortResultTable = sortrows(allResultTable,1);
        hypers_learned = sortResultTable(1,3:end)';
%         results.nlml = sortResultTable(1,1);
        results.numofitr = sortResultTable(1,2);
    else
        hypers_learned = gpmodel.hyperparams(1,:)';
    end

    [nlml, ~, alpha_kron, prodKs] = loglikefunc( hypers_learned );
    results.nlml = nlml;
    
    if(predictionFlag)
        
        predictfunc = @(t) feval(predfunc, xstar, t,  input, gpmodel, alpha_kron, prodKs);

        %learnexectime =toc
%         disp(nlml);
        


        % use meshgid to create a locations for interpolation
        % we interpolate also observed locations for denoising

        % allocate memory for variance of prediction
        %     Iout = zeros(size(IoutSize));
        %     Vout = zeros(size(IoutSize));

        % perform prediction using covMatern covariance function
        if(voutFlag)
            [mu_f, var_f] = predictfunc(hypers_learned);
        else
            [mu_f] = predictfunc(hypers_learned);
%             [mu_f] = gpgrid_predict_fullkron_denoise(hypers_learned, input, gpmodel, alpha_kron, prodKs);

            var_f = 0;
        end
        Iout = mu_f;
        Vout = var_f;
        
%         gpmodel.noise_struct.var = ones(size(gpmodel.noise_struct.var ));

    end
end

% make sure to unlock the prodcovfunc since it was locked for the use of the presistent
% variable 
% if(mislocked(prodcovfunc))
    munlock(prodcovfunc)
% end

end