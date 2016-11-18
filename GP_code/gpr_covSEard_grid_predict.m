function [mu_f, std_f] = gpr_covSEard_grid_predict(xstar, logtheta, xgrid, ygrid, alpha)

%Fast version of GP prediction ALL points lie on a grid (with G grid points per dim)
%This version also attempts to do everything in linear memory complexity
%NOTE: Only works with axis-aligned covariance functions (e.g. covSEard) + spherical noise (e.g. covNoise)

%xgrid : cell array of per dimension grid points
%ygrid : targets corresponding to inputs implied by xgrid
%xstar : new input point where we want to make predictions

D = length(xgrid); %number of dimensions
N = prod(cellfun(@length, xgrid));
assert(D > 0);
assert(size(xstar,1) == D);
assert(size(ygrid,1) == N);

%tic;

Kstars = cell(D,1);
Qs = cell(D,1);
QTs = cell(D,1);
for d = 1:D
    
    xg = xgrid{d}'; 
    K_d = feval('covSEardOLD', [logtheta(d); logtheta(D+1)/D], xg);
    [jj, Kstar_d] = feval('covSEardOLD', [logtheta(d); logtheta(D+1)/D], xg, xstar(d));
    [Q,V] = eig(K_d);
    beta = Q'*Kstar_d;
    V = diag(V);
    if d == 1
        V_kron = V;
        beta_kron = beta;
        Kstar_kron = Kstar_d;
    else
        V_kron = kron(V_kron, V); %this is a vector so still linear in memory
        beta_kron = kron(beta_kron, beta);
        Kstar_kron = kron(Kstar_kron, Kstar_d);
    end
    % do not need to store here, but useful for debugging
    Kstars{d} = Kstar_d;
    Qs{d} = Q;
    QTs{d} = Q';
    
end

noise_var = exp(logtheta(D+2))^2;
V_kron = V_kron + noise_var*ones(N,1);
%printf('Inverse condition number = %e\n', min(V_kron)/max(V_kron));
mu_f = real(Kstar_kron'*alpha);
Kss = feval('covSEardOLD', logtheta, xstar');
std_f = sqrt(Kss - sum((beta_kron.^2)./V_kron));

%toc
    

