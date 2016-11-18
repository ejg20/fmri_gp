function [hypers_learned, trnlml, Iout_z, Iout, Vout, filename] = run_gp_grid(gp_input, gpmodel, varargin)
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

Iout_z=[];
Iout=[];
Vout=[];

dft_params.wm = std(gp_input.zeromeandata);
Gs = gp_grid_size(gp_input.xgrid);
dft_params.sm = 2*Gs./gp_input.Fs;

pnames = {'params', 'xstar', 'lognoise', 'numofstarts', 'maxiteration', 'hypers_init', 'filename','predictVar'};
dflts =  {dft_params, [], -1, 50, 500, [], 'gp_grid_workspace_tmp', false};
[params, xstar, lognoise, numofstarts, maxiteration, hypers_init, filename, predictVar] ...
    = internal.stats.parseArgs(pnames, dflts, varargin{:});


bestnlml = inf;
for numoftries = 1:4    % give a few tries just if initial values are very bad and can't converge
    if(gpmodel.learn)
        if(isempty(hypers_init));
            for nrandstart=1:numofstarts 
                gpmodel.hyperparams = [make_gpsmp_hyps_initvals(gp_input, gpmodel,3,params)', lognoise];
                [hypers_learned, trnlml] = gpgrid(gp_input, gpmodel, xstar, 1);
                disp(trnlml)
              
                if(bestnlml > trnlml.nlml)
                    bestnlml = trnlml.nlml;
                    bestinit = hypers_learned;
                end
                %             nrandstart
            end
            gpmodel.hyperparams = bestinit';
        else
            gpmodel.hyperparams = (hypers_init(:))';
        end
        
%         %%%%% TAKE AWAY %%%%
%         global cbuff
%         cbuff = [];
%         predictVar = false;
%         gp_input.P=2;
%         %%%%%%%%%%%%%%%%%%%%
        
        
        [hypers_learned, trnlml] = gpgrid(gp_input, gpmodel, xstar, maxiteration);
        hypers_init = hypers_learned;
    else
        if(isempty(hypers_init))
            error('empty hyperparameter vector');
        end
        gpmodel.hyperparams = (hypers_init(:))';
        break;
    end
    
    if(trnlml.numofitr>min(maxiteration,20))    % valid optimization if num of iterations excided 5% of allowed iterations
        break;
    end
end

save(filename);

if(~isempty(xstar))
    gpmodel.hyperparams = (hypers_init(:))';
    gpmodel.learn = false;
    if(predictVar)
        [hypers_learned, trnlml2, Iout_z, Vout] = gpgrid(gp_input, gpmodel, xstar, maxiteration);
    else
        Vout = [];
        [hypers_learned, trnlml2, Iout_z] = gpgrid(gp_input, gpmodel, xstar, maxiteration);
    end
    Iout = Iout_z+gp_input.meandata;
end



save(filename);