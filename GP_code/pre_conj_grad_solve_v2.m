function [x, numofitr, rhoratio] = pre_conj_grad_solve_v2(prodKs, noise, input_data, index_to_N, C, numIter, threshold, x0)
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
%   prodKs
%       .Qs : per-dimension eigenvector matrix (kronecker component)
%       .V_kron : eigenvalue vector (kronecker component)
%   noise : noise vector (corresponds to diagonal noise)
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

c1 = 1;
c2 = 1;


if(nargin < 7)
    threshold = 1e-5; %used to assess convergence of conjugate gradient solver
end

Z = length(prodKs);
P = length(prodKs(1).Ks);
N = 1;
for p = 1:P
    N=N*size(prodKs(1).Ks{p},1);
end

if(length(noise) ~= N)
    error('Noise vector not equal N');
end

noise=noise(:);
n = length(input_data);
if(length(index_to_N) ~=  n)
    error('Index to N does not equal n');
end

P = zeros(N,1);
b = input_data;

if(nargin < 8 || isempty(x0))
    x = zeros(n,1); %need to add option for initial guess!
    r = b; %initial residual -- used to assess convergence
else
    x = x0;
    P(index_to_N) = x0;
    Ap = noise(index_to_N).*x0;
    % for each product kernel perform fast computations using Kronecker
    for z = 1:Z
        Apz = kron_mv(prodKs(z).Ks, P);
        %         Qtp = kron_mv(prodKs(z).QTs, P);
        %         maxV = max(prodKs(z).V_kron);
        %         VQtp = prodKs(z).V_kron.*Qtp.*(prodKs(z).V_kron > 0.1*maxV);
        %         Apz = kron_mv(prodKs(z).Qs, VQtp);
        Ap = Ap + Apz(index_to_N);
    end
    
    r = b-Ap;
end

Cr = C.*r;

k = 0;
rho_b = sqrt(sum(b.^2));
rho_r = Inf;

while ((rho_r > (threshold*rho_b)) && (k < numIter))
    %move index
    k = k+1;
    if k > 1
        p_prev = p;
        r_prev_prev = r_prev;
        Cr_prev_prev = Cr_prev;
    end
    r_prev = r;
    Cr_prev = Cr;
    x_prev = x;
    
    %iteration core
    if k == 1
        p = Cr_prev;
    else
        beta = c2*(Cr_prev'*r_prev)/(Cr_prev_prev'*r_prev_prev); % Fletcher–Reeves
        %         beta = (Cr_prev'*(r_prev-r_prev_prev))/(Cr_prev_prev'*r_prev_prev); % Polak–Ribière
        p = Cr_prev + beta*p_prev;
    end
    
    % Compute (Q1*V1*Q1' + Q2*V2*Q2' +.. QZ*VZ*QZ' + diag(noise))*p
    P(index_to_N) = p;
    Ap = noise(index_to_N).*p;
    % for each product kernel perform fast computations using Kronecker
    for z = 1:Z
        Apz = kron_mv(prodKs(z).Ks, P);
        %          Qtp = kron_mv(prodKs(z).QTs, P);
        %          maxV = max(prodKs(z).V_kron);
        %           VQtp = prodKs(z).V_kron.*Qtp.*(prodKs(z).V_kron > 0.1*maxV);
        %            Apz = kron_mv(prodKs(z).Qs, VQtp);
        Ap = Ap + Apz(index_to_N);
    end
    
    pAp = p'*Ap; %for numerical stability
    if(pAp == 0)
        alpha = 0;
    else
        alpha = c1*(Cr_prev'*r_prev)/pAp;
    end
    x = x_prev + alpha*p;
    r = r_prev - alpha*Ap;
%     plot(r); ylim([-1,1]); drawnow; max(r)
    Cr = C.*r;
    rho_r = sqrt(r'*r);
    rho_buff(k) = rho_r;
    alpha_buff(k) = alpha;
    
%     if(mod(k,500) == 0)
%         kvec = k-40:k;
% %         m = (kvec*rho_buff(kvec)')/sum(kvec.^2)
%         m = polyfit(1:41,rho_buff(kvec),1);
%         if(m > -0.0001)
% %                 stop = 1;
%             break;
%         end
%     end
%     c1 = 0.9995^k;
end

numofitr = k;
rhoratio = rho_r/rho_b;




