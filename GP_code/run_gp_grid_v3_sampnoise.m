function [hypers_learned, trnlml, Iout_z, Iout, Vout, filename] = run_gp_grid_v2(gp_input, gpmodel, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% run_gp_grid
%
% Wrapper function for running gp_grid efficiently
%
% Usage: [hypers_learned, trnlml, Iout_z, Iout, Vout, filename] = run_gp_grid(gp_input, gpmodel, varargin)
%
% Inputs:
%       gp_input       gp_grid_input_class
%       gpmodel     gp_grid_gpmodel_class
%       varargin:
%           'params' -      initialization parameters
%                           (default dft_params.wm = std(gp_input.zeromeandata);
%                                    dft_params.sm = 2*Gs./gp_input.Fs;)
%           'xstar' -       prediction locations (default [])
%           'lognoise' -    log of the Gaussian noise hyperparametr (default -1)
%           'numofstarts' - number of random restart, looking for the optimal place to start (default 50)
%           'maxiteration' - number of maximize iteration (default 500)
%           'hypers_init' - if only want to predict and not learn (default [])
%           'predictVar' -  flag for predicting the test variance
%           'filename' -    specify a name to save workspace (default 'gp_grid_workspace_tmp')
%
% Outputs:
%       hypers_learned  learned hyperparameters
%       trnlml          gp_grid test statistics
%       Iout_z          normalized predictions
%       Iout            predictions after addition of training mean
%       Vout            prediction variance (very slow)
%       filename        saved filed name containing workspace
%
%
% Note:
% To use the package you must first install gpml-matlab.
% The package can be found in www.gaussianprocess.org/gpml/. The last
% version tested was gpml-matlab-v3.2-2013-01-15
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if(nargin < 2)
    error('not enough parameters')
end

dft_params.wm = std(gp_input.zeromeandata);
Gs = gp_grid_size(gp_input.xgrid);
dft_params.sm = 2*Gs(:)./gp_input.Fs(:);

pnames = {'params', 'xstar', 'lognoise', 'numofstarts', 'maxiteration', 'hypers_init', 'filename','predictVar'};
dflts =  {dft_params, [], -1, 50, 500, [], [], false};
[params, xstar, lognoise, numofstarts, maxiteration, hypers_init, filename, predictVar] ...
    = internal.stats.parseArgs(pnames, dflts, varargin{:});

Iout_z=[];
Iout=[];
Vout=[];


for numoftries = 1:4    % give a few tries just if initial values are very bad and can't converge
    if(gpmodel.learn)
        if(isempty(hypers_init));
            bestnlml = inf;
            bestinit = [];
            bestlognoise = lognoise;
            %if specified a distribution over the noise initial value
            if(~isempty(params.noise.gamma))
                
                for nrandstart=1:numofstarts
                    
%                     phat=params.noise.gamma;
                    phat = [1.3064  85.205];
                    %use a gamma distribution for initial rand value for
                    %the noise.
                    lognoise =log(sqrt(abs(gamrnd(phat(1),phat(2)))))*ones(size(lognoise));
%                     disp(exp(2*lognoise(1)));
                    
                    gpmodel.hyperparams = [make_hyps_initvals_v2(gp_input, gpmodel,3,params)', lognoise];
                    
                    [hypers_learned, trnlml] = gpgrid_v2(gp_input, gpmodel, xstar, 1, 'checkinit', true, 'params',params(1));
                    disp(trnlml)
                    %                 disp(trnlml.nlml_lik-trnlml.nlml_lik)
                    if(bestnlml > trnlml.nlml_lik)
                        bestnlml = trnlml.nlml_lik;
                        bestinit = hypers_learned;
                        bestlognoise = lognoise;
                    end
                    %             nrandstart
                end
                
            end
            for nrandstart=1:numofstarts
%                 disp(exp(2*bestlognoise(1)));
                gpmodel.hyperparams = [make_hyps_initvals_v2(gp_input, gpmodel,3,params)', bestlognoise];
                
                [hypers_learned, trnlml] = gpgrid_v2(gp_input, gpmodel, xstar, 1, 'checkinit', true, 'params',params(1));
                disp(trnlml.nlml_lik)
             
                if(bestnlml > trnlml.nlml_lik)
                    bestnlml = trnlml.nlml_lik;
                    bestinit = hypers_learned;
                end
               
            end
            if(isempty(bestinit))
                continue;
            end
            gpmodel.hyperparams = bestinit(:)';
        else
            gpmodel.hyperparams = (hypers_init(:))';
        end
        
        for numoftriesin = 1:4
            disp(exp(2*gpmodel.hyperparams(end)));
            [hypers_learned, trnlml] = gpgrid_v2(gp_input, gpmodel, xstar, maxiteration,'params',params(1));
            gpmodel.hyperparams = (hypers_learned(:))';
            if(trnlml.numofitr>min(maxiteration,20))    % valid optimization if num of iterations excided 5% of allowed iterations
                hypers_init = hypers_learned;
                break;
            end
        end
        
    else
        if(isempty(hypers_init))
            error('empty hyperparameter vector');
        end
        gpmodel.hyperparams = (hypers_init(:))';
        break;
    end
    
    if(trnlml.numofitr>min(maxiteration,20))    % valid optimization if num of iterations excided 5% of allowed iterations
        hypers_init = hypers_learned;
        break;
    end
end

if(~isempty(filename))
    save(filename);
end

if(~isempty(xstar))
    gpmodel.hyperparams = (hypers_init(:))';
    gpmodel.learn = false;
    
    
    if(predictVar)
        [hypers_learned, trnlml, Iout_z, Vout] = gpgrid_v2(gp_input, gpmodel, xstar, maxiteration,'params',params(1));
    else
        Vout = [];
        [hypers_learned, trnlml, Iout_z] = gpgrid_v2(gp_input, gpmodel, xstar, maxiteration,'params',params(1));
    end
    
    Iout = Iout_z+gp_input.meandata;
end
if(~isempty(filename))
    save(filename);
end