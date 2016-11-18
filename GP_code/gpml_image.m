%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Elad Gilboa
% 2013
%
% gpgrid_dn()
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
function [logtheta_learned, Iout, Vout] = gpml_image(input, x, IoutSize, noise, xstar, InitParamSet, cov)

% % allocate memory
% Iout = zeros(IoutSize(:)');
% If only one output than only return learned hyperparameters
% Otherwise, also perform prediction for Iout
if(nargout > 1)
    predictionFlag = true;
    if(nargout > 2)
        voutFlag = true;
    else
        voutFlag = false;
    end
else
    predictionFlag = false;
end
% y is a column stack of Isub
y = input.data(:);
D=2;

% initial points for the hyperparameters optimization, hoping to hit global
% optimum
ellscoef = InitParamSet.l;
sfcoef = InitParamSet.sf;
if(isfield(InitParamSet,'sn') && ~isempty(InitParamSet.sn))
    sncoef = InitParamSet.sn;
    learnNoiseFlag = true;
else
    sncoef = 0;
    learnNoiseFlag = false;
end

allResultTable = zeros(length(sncoef)*length(sfcoef)*length(ellscoef),1+2*(D+1+learnNoiseFlag));

testnum = 1;
if(~isfield(InitParamSet,'learn') || InitParamSet.learn==true)
    for sni = 1:length(sncoef)
        if(learnNoiseFlag)
            sn = sncoef(sni);
        else
            sn = [];
        end
        for sfi = 1:length(sfcoef)
            sf = sfcoef(sfi);
            for ellsi = 1:length(ellscoef)
                
                % initial guess lengthscales for all dimensions are equal
                ells = ellscoef(ellsi)*ones(D,1);
                
                % optimize hyperparameters using covariance function
%                 [logtheta_learned fx] = minimize(logtheta_init, 'gpr_cov_grid_dn', -100, Xgrid, input, noise, cov);
                likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
                hyp2.cov = log([ells ; sf]);    
                hyp2.lik = log(sn);
tic
                [logtheta_learned fx] = minimize(hyp2, @gp, -50, @infExact, [], cov, likfunc, x, y);
toc
                %%

                allResultTable(testnum,:) = [fx(end),logtheta_learned',logtheta_init'];
                testnum = testnum +1;
            end
            
        end
    end
    
    sortResultTable = sortrows(allResultTable,1);
    logtheta_learned = sortResultTable(1,2:1+(D+1+learnNoiseFlag));
    
else
    logtheta_learned = log([InitParamSet.l(:);InitParamSet.sf(:);InitParamSet.sn]);
end

% disp(['sf learned = 2^',num2str(imin-1),' ells learned = 2^',num2str(imin_learnedtt(imin)-1)]);
% disp((logtheta_learned)./log(2))
% disp(exp(logtheta_learned))



% pause

if(predictionFlag)
    % call gpr_cov_grid_dn one more time with optimal hyperparameters
    % [nlml, dnlml, alpha_kron, Qs, V_kron] = gpr_cov_grid_dn(logtheta_learned, Xgrid, input, noise, cov);
    [nlml, dnlml, alpha_kron, Qs, V_kron] = gpr_cov_grid_dn_LRApprox(logtheta_learned, Xgrid, input, noise, cov);
    
    %learnexectime =toc
    disp(nlml);
    % if the noise model was spherical and the hyperparamter was learned, then
    % get the new noise matrix
    if(learnNoiseFlag == true)
        sn = exp(logtheta_learned(D+2));
        noise = sn*noise;
    end
    

    [Iout] = gp(hyp2, @infExact, [], covfunc, likfunc, x, y, xstar);
    Vout = 0;

end
% keyboard


end