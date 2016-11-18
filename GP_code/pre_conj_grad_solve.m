function [x, numofitr] = pre_conj_grad_solve(Qs, V, noise, input_data, input_index_to_N, C, numIter, threshold, x0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% pre_conj_grad_solve()
%
% Fast version of PCG using kronecker structure
%
%
% Usage: [pre_conj_grad_solve(Qs, V, noise, input_data, input_index_to_N, C, numIter, threshold)
%
% Inputs:
%       Qs : per-dimension eigenvector matrix (kronecker component)
%       V_kron : eigenvalue vector (kronecker component)
%       noise : noise vector (corresponds to diagonal noise)
%
% Outputs:
%     x             output vector x=A^-1*y
%     numofitr      number of PCG iterations
%
%
% Note:
% Taken almost verbatim from Golub and van Loan pp. 527
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if(nargin < 8)
    threshold = 1e-5; %used to assess convergence of conjugate gradient solver 
end

D = length(Qs);
QTs = cell(D,1);
N = 1;
for d = 1:D
    QTs{d} = Qs{d}';
    N=N*size(Qs{d},1);
end

noise=noise(:);
n = length(input_data);
P = zeros(N,1);
b = input_data;
index_to_N = input_index_to_N;

if(nargin < 9 || isempty(x0))
    x = zeros(n,1); %need to add option for initial guess!
    r = b; %initial residual -- used to assess convergence
else
    x = x0;
    P(index_to_N) = x0; 
    Ap = kron_mv(QTs, P);
    Ap = Ap.*V;
    Ap = kron_mv(Qs, Ap);
    Ap = Ap(index_to_N) + noise(index_to_N).*x0;
    r = b-Ap;
end

z = C.*r;

k = 0;
rho_b = sqrt(sum(b.^2));
rho_r = Inf;

while ((rho_r > (threshold*rho_b)) && (k < numIter))
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
    
	% Compute (Q*V*Q' + diag(noise))*p
    P(index_to_N) = p; 
    Ap = kron_mv(QTs, P);
    Ap = Ap.*V;
    Ap = kron_mv(Qs, Ap);
    Ap = Ap(index_to_N) + noise(index_to_N).*p;
    pAp = p'*Ap; %for numerical stability
    if(pAp == 0)
        alpha = 0;
    else
        alpha = (z_prev'*r_prev)/pAp;
    end
	x = x_prev + alpha*p;
	r = r_prev - alpha*Ap;
    z = C.*r;
    rho_r = sqrt(r'*r);

end

numofitr = k;

	
	
	 
	
