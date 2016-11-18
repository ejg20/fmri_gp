function [nlml, dnlml] = gpr_covSEard(logtheta, xgrid, y)

%Usual way of doing stuff

%tic;
covfunc = {'covSum', {'covSEardOLD'}};
x = cartprod(xgrid);
N = size(x,1);
K = feval(covfunc{:}, logtheta, x);    % compute training set covariance matrix
L = chol(K, 'lower'); clear K;
alpha = L'\(L\y);

nlml = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*N*log(2*pi);

dnlml = zeros(size(logtheta));       % set the size of the derivative vector
W = L'\(L\eye(N))-alpha*alpha'; clear L;                % precompute for convenience
for i = 1:length(dnlml)
    dnlml(i) = sum(sum(W.*feval(covfunc{:}, logtheta, x, i)))/2;
end
%toc