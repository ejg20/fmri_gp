function [hypers_learned, gp_input, gpmodel, lognoise, noise_std_mat] = ...
    fmriGP_learn_sampnoise(ytr, mask, lambda, s2, sampleinitnoise, varargin)


pnames = {'nolearn', 'Fs', 'spacetimesplitdim'};
dflts =  {false, [], 3};
[nolearn, Fs, spacetimesplitdim] = internal.stats.parseArgs(pnames, dflts, varargin{:});

hypers_learned = [];

sz=size(ytr);
if(length(sz)< 5)   %HACK!!
    sz = [sz,1,1,1];
end


if(isempty(Fs))
    if(length(size(ytr)) == 5)
        Fs = [3e-3, 3e-3, 3e-3, 2, 1].^(-1);
    else
        Fs = [3e-3, 3e-3, 3e-3, 2].^(-1);
    end
end
if(~isempty(spacetimesplitdim))
    repmaskdim = [ones(1,spacetimesplitdim),prod(sz((spacetimesplitdim+1):end))];
    gp_input = gp_grid_input_class(ytr, find(repmat(mask,repmaskdim)), Fs);
else
    gp_input = gp_grid_input_class(ytr, find(repmat(mask,[1 1 1 prod(sz(4:end))])), Fs);
end



noise_struct = gp_grid_noise_class(ones(size(ytr)),gp_input.index_to_N);
noise_struct.learn = true;

if(length(size(ytr)) == 5)
    covs{1} = {{'covSM1D',1},{'covSM1D',20},{'covSM1D',1},{'covSM1D',1},{'covSM1D',1}};
elseif(length(size(ytr)) == 2)
    covs{1} = {{'covSM1D',1},{'covSM1D',1},{'covSM1D',1},{'covSM1D',1}};
else
    covs{1} = {{'covSM1D',20},{'covSM1D',1},{'covSM1D',1},{'covSM1D',1}};
end

gpmodel = gp_grid_gpmodel_class();
gpmodel.cov = covs;
gpmodel.noise_struct = noise_struct;
gpmodel.hyps_in_d  = make_hyps_in_d_v2(covs);
gpmodel.learn = true;


xstar = [];
params.wm = std(gp_input.zeromeandata);
% Gs = gp_grid_size(gp_input2.xgrid);
% params.sm = 2*Gs(:)./gp_input2.Fs;
if(length(size(ytr)) == 5)
%     params.sm = [1, 20, 0.05, 0.05, 0.05];
params.sm = [1, 20, 0.05, 0.05, 0.05];
else
    params.sm = [20, 0.05, 0.05, 0.05];
end

params.prodcovfunc = 'gp_grid_prodcov_v2_hetero';
params.predfunc = 'gpgrid_predict_fullkron_denoise';


% TRY HETRO NOISE. neet to change it for resting state data!!


noise_std_mat = sqrt(s2);

if(sampleinitnoise)
    [phat] = gamfit(s2);
    params.noise.gamma=phat;
else
    params.noise.gamma=[];
end

if(size(noise_std_mat,1) == 1)
    %homogenous noise
    gpmodel.noise_struct.groups = {gp_input.index_to_N};
elseif size(noise_std_mat,1) == sum(mask(:))
    %heterogeneouse noise
%     if(length(sz)>4)
%         indxmat = reshape(1:gp_input.get_N,prod(sz(1:3)),prod(sz(4:end)));
%     else
%         indxmat = 1:gp_input.get_N;
%     end
    indxmat = reshape(1:gp_input.get_N,prod(sz(1:spacetimesplitdim)),...
        prod(sz((spacetimesplitdim+1):end)));
    indxmat = indxmat(mask(:),:);
    if size(noise_std_mat,2) == 1
        %homogeneous per voxel
        gpmodel.noise_struct.groups = mat2cell(indxmat,ones(1,size(indxmat,1)),size(indxmat,2));
    else
        %heteroscedastic per voxel (also per time)
        gpmodel.noise_struct.groups = num2cell(indxmat(:));
    end
else
end


lognoise = log(noise_std_mat(:))';

% if lambda is not 0 then use a Gaussian prior for the noise, if 0 then use
% an uninformative prior (=no prior)
if(lambda)
    gpmodel.logpriorfuns{1}.func = @(t) gp_grid_Gaussian(t, lambda, noise_std_mat(:).^2, 2);
    gpmodel.logpriorfuns{1}.indices = length(make_hyps_initvals_v2(gp_input, gpmodel,3,params))+(1:length(lognoise));
end

if(nolearn)
    return;
end
[hypers_learned, ~, ~, ~, ~, ~] = run_gp_grid_v3_sampnoise(gp_input, gpmodel,'params',params,'xstar',xstar,...
    'lognoise',lognoise,'numofstarts',30,'maxiteration',200);
