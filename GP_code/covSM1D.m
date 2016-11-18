function [K hlpvars] = covSM1D(Q, hyp, x, z, derivnum, hlpvars) %need to
% change notation to work with gpml!!

% Spectral Mixture covariance function for a signal component. The covariance function is:
%
%   k(tau) = sum_{q=1}^Q { w_q^2 * exp(-2*pi^2*tau^2*s2_q) * cos(2*pi*tau*mu_q) }
%
% with tau = x-x', and Q is number of Gaussians.
% The weights w_q specify the relative contribution of each mixture
% component. The inverse means 1/mu_q are the componenet periods, and the
% inverse standard deviation 1/sqrt(s2_q) are lengthscales.
% Assume column vectors
% The hyperparameters are:
%
% hyp = [ log(w_1)
%         log(w_2)
%            .
%         log(w_Q)
%         log(mu_1)
%         log(mu_2)
%            .
%         log(mu_Q)
%         log(sqrt(s2_1))
%         log(sqrt(s2_2))
%            .
%         log(sqrt(s2_Q))]
%
% hyp size 3*Q.
%
% Copyright Elad Gilboa, 2013-7-10.
% Based on Gaussian spectral mixture covariance function by
% Andrew Gordon Wilson, 2 Feb 2013
% http://mlg.eng.cam.ac.uk/andrew
%
% See also COVFUNCTIONS.M.

if nargin < 3, K = num2str(3*Q); return; end          % report number of hyperparameters
if nargin < 4, z = []; end                                 % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

if(mod(length(hyp),3) >0 ), error('number of parameters must be 3Q'); end % not a valid number of parameters

% get hyperparameters
if(Q ~= length(hyp)/3)
    error('number of hyperparameters does not equal 3Q');
end
w2 = exp(2*hyp(1:Q));
mu = exp(hyp(Q+(1:Q)));
s2 = exp(2*hyp(2*Q+(1:Q)));


if nargin>4 && exist('hlpvars','var')  && length(hlpvars) == 2   % check if distance matrices were given as input to avoid unecessary computations
    S2 = hlpvars{1};
    S = hlpvars{2};
else
    if dg                                                               % vector kxx
                    % dont really need these calculations 
%         S2 = zeros(size(x,1),1);
%         S = S2;
    elseif xeqz                                       % symmetric matrix Kxx
        S2 = sq_dist(x');
        S = sqrt(S2);
    else                                          % cross covariances Kxz
        S2 = sq_dist(x',z');
        S = sqrt(S2);
    end
end

if nargin<=4
    % precompute squared distances
    if dg                                                               % vector kxx
        K = ones(size(x,1),1)*sum(w2);
    elseif xeqz                                       % symmetric matrix Kxx
        K = zeros(length(x));                       % zero matrix length(x) by length(x)
        for q = 1:Q
            C_q = exp(-2*pi^2*S2*s2(q)).*cos(2*pi*S*mu(q));
            K = K + w2(q)*C_q;
        end
    else                                          % cross covariances Kxz
        K = zeros(length(x),length(z));                       % zero matrix length(x) by length(x)
        for q = 1:Q
            C_q = exp(-2*pi^2*S2*s2(q)).*cos(2*pi*S*mu(q));
            K = K + w2(q)*C_q;
        end
    end
    
end

if nargin>4                                                      % derivatives
    effectivederivnum=mod(derivnum-1,Q)+1;
    if derivnum<=Q                                                  % weight parameters
        if dg
            K = ones(size(x,1),1)*w2(effectivederivnum)*2;
        else
            % The important thing here are the S2, S matrices which
            % were calculated beforehand
            K = exp(-2*pi^2*S2*s2(effectivederivnum)).*cos(2*pi*S*mu(effectivederivnum))*w2(effectivederivnum)*2;
        end
    elseif derivnum<=2*Q                                            % mean parameter
        if dg
            K = zeros(size(x,1),1);
        else
            K =  - w2(effectivederivnum)*exp(-2*pi^2*S2*s2(effectivederivnum)).*sin(2*pi*S*mu(effectivederivnum))*2*pi.*S*mu(effectivederivnum);
        end
    elseif derivnum<=3*Q                                            % var parameter
        if dg
            K = zeros(size(x,1),1);
        else
            K = -2*w2(effectivederivnum)*exp(-2*pi^2*S2*s2(effectivederivnum)).*cos(2*pi*S*mu(effectivederivnum))*2*pi^2.*S2*s2(effectivederivnum);
        end
    else
        error('Unknown hyperparameter')
    end
end

if(exist('S2','var'))
    hlpvars{1} = S2;
    hlpvars{2} = S;
else
    hlpvars = {};
end

end


