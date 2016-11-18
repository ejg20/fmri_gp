function [nlml, dnlml, alpha_kron] = gpr_covMaterniso_grid(logtheta, xgrid, y)
logtheta
N = prod(cellfun(@length, xgrid));
D = length(xgrid);
if size(y,1) ~= N
    error('Invalid vector of targets, quitting...');
end
if length(logtheta) ~= D+2
    error('Error: Number of parameters do not agree with covariance function!')
end

Ks = cell(D,1);
Qs = cell(D,1);
QTs = cell(D,1);
for d = 1:D
    
    xg = xgrid{d};
    if size(xg,2) > size(xg,1)
        xg = xg';
    end
    K_kron = feval('covSEardOLD', [logtheta(d); logtheta(D+1)/D], xg); %TODO: Toeplitz
    Ks{d} = K_kron;
    [Q,V] = eig(K_kron); %TODO: Toeplitz
    Qs{d} = Q;
    QTs{d} = Q';
    V = diag(V);
    if d == 1
        V_kron = V;
    else
        V_kron = kron(V_kron, V); %this is a vector so still linear in memory
    end
    
end

noise_var = exp(logtheta(D+2))^2;
V_kron_noise = V_kron + noise_var*ones(N,1)+1e-10;   %epsilon for computational stability
%fprintf('Inverse condition number = %e\n', min(V_kron)/max(V_kron));
alpha_kron = kron_mv(QTs, y(1:N));
alpha_kron = alpha_kron./V_kron_noise;
alpha_kron = kron_mv(Qs, alpha_kron);
%alpha_k = conj_grad_solve(Qs, V_kron, noise_var*ones(N,1), y);

logdet_kron = sum(log(V_kron_noise));
nlml = real(0.5*(alpha_kron'*y + logdet_kron + N*log(2*pi)));

%Now for the derivatives
dnlml = zeros(size(logtheta));

%lengthscales
ell = exp(logtheta(1:D));                         
for i = 1:D
    dC = cell(D,1);
    dK = cell(D,1);
    for d = 1:D
        if d == i
            dK_kron = Ks{d} .* sq_dist(xgrid{i}/ell(i));
        else
            dK_kron = Ks{d};
        end
        dK{d} = dK_kron;
        dC{d} = QTs{d}*dK_kron'*Qs{d};
        if d == 1
            diag_Z = diag(dC{d});
        else
            diag_Z = kron(diag_Z, diag(dC{d}));
        end
    end
    dnlml(i) = 0.5*(sum(diag_Z./V_kron_noise) - alpha_kron'*kron_mv(dK, alpha_kron));
end
%amplitude & noise
noise = exp(2*logtheta(D+2));
dC = cell(D,1);
dN = cell(D,1);
for d = 1:D
    dC{d} = QTs{d}*Ks{d}'*Qs{d};
    dN{d} = QTs{d}*Qs{d};
    if d == 1
        diag_Z = diag(dC{d});
        diag_N = diag(dN{d});
    else
        diag_Z = kron(diag_Z, diag(dC{d}));
        diag_N = kron(diag_N, diag(dN{d}));
    end
end
dnlml(D+1) = sum(diag_Z./V_kron_noise) - alpha_kron'*kron_mv(Ks, alpha_kron);
dnlml(D+2) = noise*(sum(diag_N./V_kron_noise) - alpha_kron'*alpha_kron);
dnlml = real(dnlml);
    


