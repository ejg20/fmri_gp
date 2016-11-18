function [x, residuals] = pre_conj_grad_solve_testThreshold(Qs, V, noise, input, C, numIter,epsilon)

%Qs : per-dimension eigenvector matrix (kronecker component)
%V_kron : eigenvalue vector (kronecker component)
%noise : noise vector (corresponds to diagonal noise)

%  = 1e-7; %used to assess convergence of conjugate gradient solver 

D = length(Qs);
QTs = cell(D,1);
N = 1;
for d = 1:D
    QTs{d} = Qs{d}';
    N=N*size(Qs{d},1);
end

%Taken almost verbatim from Golub and van Loan pp. 527
% numIter = 30000;
%  numIter = 3*length(b);
b = input.data;
index_to_N = input.index_to_N;
n = length(b);
k = 0;
r = b; %initial residual -- used to assess convergence
z = C.*r;
x = zeros(n,1);
rho_b = sqrt(sum(b.^2));
rho_r = Inf;
residuals = zeros(numIter,1);
P = zeros(N,1);
while ((rho_r > (epsilon*rho_b)) && (k < numIter))
	%move index
	k = k+1;
	if k > 1
		p_prev = p;
		r_prev_prev = r_prev;
        z_prev_prev = z_prev;
	end
	r_prev = r;
    z_prev = z;
	x_prev = x;
	
	%iteration core
	if k == 1
		p = z_prev;
	else
		beta = (z_prev'*r_prev)/(z_prev_prev'*r_prev_prev);
		p = z_prev + beta*p_prev;
	end
	%Compute (Q*V*Q' + diag(noise))*p
    P(index_to_N) = p; 
    Ap = kron_mv(QTs, P);
    Ap = Ap.*V;
    Ap = kron_mv(Qs, Ap);
    Ap = Ap(index_to_N) + noise(index_to_N).*p;
    
	alpha = (z_prev'*r_prev)/(p'*Ap);
	x = x_prev + alpha*p;
	r = r_prev - alpha*Ap;
    z = C.*r;
    rho_r = sqrt(r'*r);
% 	fprintf('Residual = %3.3f\n', rho_r);
    residuals(k) = rho_r;
    
    if k == numIter
        figure(33); plot(residuals); drawnow;
    end
end

residuals = residuals(1:k);

% fprintf('Conjugate gradient solver converged in %i iterations.\n',k); 
	
	
	 
	
