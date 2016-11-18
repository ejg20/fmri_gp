function [nlml, dnlml] = gp_grid_Gaussian(hypvec, lambda, mu, pow)
% calculate Gauss distribution
% hypvec contains log(theta) hyperparameters
%
%
% SANITY CHECK 
% gp_grid_Gaussian - ?
% checkgrad('gp_grid_Gaussian', hypers_init, 1e-3, mu, pow)
%
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(lambda==0)
    nlml=0;
    dnlml=zeros(size(hypvec));
    return
end
Q = length(hypvec);
theta = exp(pow*hypvec);
nlml = lambda*sum((theta(:)-mu(:)).^2) - Q/2*log(lambda/pi); % negative log likelihood

% Derivatives
dnlml =  2*lambda*(theta(:)-mu(:)).*theta(:)*pow; %dnlml/dlogtheta = dnlml/dtheta*dtheta/dlogtheta  
% disp(theta(1:3));
return
