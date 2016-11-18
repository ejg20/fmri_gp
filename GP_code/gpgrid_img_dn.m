%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Elad Gilboa
% 2013
%
% gpgrid_img_dn()
%
% Fast version of GP prediction ALL points lie on a grid (with G grid points per dim).
% This version also attempts to do everything in linear memory complexity.
% gpgrid_img_dn does not consider spherical noise as a hyperparameter to be learned,
% but as input dependent noise specified by the user.
%
%
% Usage: [logtheta_learned, Iout, Vout] = gpgrid_img_dn(Isub, Xgrid, IoutSize, noise, mask, InitParamSet, cov);
%        [logtheta_learned] = gpgrid_img_dn(Isub, Xgrid, ver_size, hor_size, noise, mask, InitParamSet, cov);
%
% Inputs:
%     Isub          the sub image to be interpolated
%     Xgrid         cell array of per dimension grid points
%     IoutSize      [ver_size hor_size] size of output image
%     noise         noise variance of Isub observations
%     mask          binary mask for output image
%     InitParamSet  structure with set of initial guesses for
%                   hyperparameters
%                   InitParamSet.l - lengthscale
%                   InitParamSet.sf - signal variation
%     cov           covariance function as in gpml-matlab
%
% Outputs:
%     logtheta_learned  learned hyperparameters
%     Iout              interpolated Image
%     Vout              GP variance (not implemented)
%
%
% Note:
% To use the package you must first install gpml-matlab.
% The package can be found in www.gaussianprocess.org/gpml/. The last
% version tested was gpml-matlab-v3.1-2010-09-27
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [logtheta_learned, Iout, Vout] = gpgrid_img_dn(Isub, Xgrid, IoutSize, noise, mask, InitParamSet, cov)

ver_size = IoutSize(1);
hor_size = IoutSize(2);
% allocate memory
Iout = zeros(ver_size,hor_size);
% If only one output than only return learned hyperparameters
% Otherwise, also perform prediction for Iout
if(nargout > 1)
    predictionFlag = true;
else
    predictionFlag = false;
end
% y is a column stack of Isub
y = Isub(:);
% D is the number of dimensions (should be 2 for Images)
D = length(Xgrid);


% initial points for the hyperparameters optimization, hoping to hit global
% optimum
ellscoef = InitParamSet.l;
sfcoef = InitParamSet.sf;
if(isfield(InitParamSet,'sn'))
    sn = InitParamSet.sn;
    learnNoiseFlag = true;
else
    sn = [];
    learnNoiseFlag = false;
end

if(~isfield(InitParamSet,'dontlearn') || InitParamSet.dontlearn==false)
    for sfi = 1:length(sfcoef)
        sf = sfcoef(sfi);
        parfor ellsi = 1:length(ellscoef)
            % initial guess lengthscales for all dimensions are equal
            ells = ellscoef(ellsi)*ones(D,1);
            
            % log of hyperparameter vector
            logtheta_init = log([ells; sf; sn]);
            
            % optimize hyperparameters using covariance function
            [logtheta_learned fx] = minimize(logtheta_init, 'gpr_cov_grid_dn', -40, Xgrid, y, noise, cov);
            %[logtheta_learned fx] = minimize([logtheta_init(1:D); logtheta_init(D+1)], 'gpr_covMaterniso_grid_dn', -40, Xgrid, y, noise);
            
            % record last (min) value and learned hyperparameters for later
            % comparison
            fxt(ellsi) = fx(end);
            logtheta_learnedt(:,ellsi) = logtheta_learned;
            
        end
        [fmin imin] = min(fxt);
        logtheta_learnedtt(:,sfi) = logtheta_learnedt(:,imin);
        fmin_learnedtt(sfi) = fmin;
        imin_learnedtt(sfi) = imin;
    end
    [fmin imin] = min(fmin_learnedtt);
    logtheta_learned = logtheta_learnedtt(:,imin);
else
    logtheta_learned = log([InitParamSet.l(:);InitParamSet.sf;sn]);
end

% disp(['sf learned = 2^',num2str(imin-1),' ells learned = 2^',num2str(imin_learnedtt(imin)-1)]);
% disp((logtheta_learned)./log(2))
% disp(exp(logtheta_learned))

% call gpr_cov_grid_dn one more time with optimal hyperparameters
[nlml, dnlml, alpha_kron, Qs, V_kron] = gpr_cov_grid_dn(logtheta_learned, Xgrid, y, noise, cov);
%learnexectime =toc

% if the noise model was spherical and the hyperparamter was learned, then
% get the new noise matrix 
if(learnNoiseFlag == true)
    s2n = exp(2*logtheta_learned(D+2));
    noise = s2n*noise;
end

if(predictionFlag)
    % use meshgid to create a locations for interpolation
    % we interpolate also observed locations for denoising
    [hor_starvec,ver_starvec] = meshgrid(1:1:ver_size,1:1:hor_size);
    xstar = [ver_starvec(:),hor_starvec(:)];
    % remove locations outside the mask
    xstar(logical(~mask(sub2ind(size(mask),xstar(:,2),xstar(:,1)))),:) = [];
    % allocate memory for variance of prediction
    Iout = zeros(ver_size,hor_size);
    Vout = zeros(ver_size,hor_size);
    % perform prediction using covMatern covariance function
    [mu_f, var_f] = gpr_cov_grid_predict_parallel(xstar, logtheta_learned, Xgrid, y, alpha_kron, noise, cov, Qs, V_kron);
% [mu_f] = gpr_cov_grid_predict_parallel(xstar, logtheta_learned, Xgrid, y, alpha_kron, noise, cov, Qs, V_kron);
    % reshape predictions to 2D image form
    Iout(sub2ind(size(Iout),xstar(:,2),xstar(:,1))) = mu_f;
    Vout(sub2ind(size(Iout),xstar(:,2),xstar(:,1))) = var_f;
    %     figure(7); imagesc(Iout); drawnow
    %     figure(8); imagesc((Vout)); drawnow
end
% keyboard


end