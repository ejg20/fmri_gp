function [ub, lb] = hadamards_ineq(KKs, Dn, index_to_N)

n = length(index_to_N);
Z = length(KKs);
P = length(KKs{1});
N = prod(cellfun(@length, KKs{1}));

sumKiiDn = 0;  % a vector to hold the sum of diag(Kii)*Dn
Kii = Dn;
for zi = 1:Z  
    Kzizi = 1;
    for pi = 1:P
        Kzipi = KKs{zi}{pi};
        Kzizi = kron(Kzizi,diag(Kzipi));    
    end
    Kii = Kii+Kzizi; 
    sumKiiDn = sumKiiDn + 2*Kzizi.*Dn;
end

invKii = zeros(N,1);
invKii(index_to_N) = Kii(index_to_N).^(-1);

Lbar = (Dn'*invKii).^2;

Ki2s = cell(P,1);
Kijs = cell(P,1);
for zi = 1:Z
    
    % calculate matrices for squared K
    for pi = 1:P
        Kzipi = KKs{zi}{pi};
        Ki2s{pi}= Kzipi.^2;
    end
    Lbar = Lbar + invKii'*kron_mv(Ki2s,invKii);
    
    % calculate cross terms
    for zj = (zi+1):Z
        for pi = 1:P
            Kzipi = KKs{zi}{pi};
            Kzjpi = KKs{zj}{pi};
            Kijs{pi} = Kzipi.*Kzjpi;

        end  
        Lbar = Lbar + 2*invKii'*kron_mv(Kijs,invKii);       
    end
    
    % calculate cross terms with diagonal noise 
    Kzizi = 1;
    for pi = 1:P
        Kzipi = KKs{zi}{pi};
        Kzizi = kron(Kzizi,diag(Kzipi));    
    end
    Lbar = Lbar + (2*Dn.*Kzizi)'*(invKii.^2);
end
        
sbar2 = Lbar/n-1;
stn = sqrt(sbar2*(n-1));
u = 1+stn;
b = 1-stn;
sdn = sqrt(sbar2/(n-1));
mu = 1-sdn;
nu = 1+sdn;

ub = log(u)+(n-1)*log(mu)+sum(log(Kii(index_to_N)));
lb = log(b)+(n-1)*log(nu)+sum(log(Kii(index_to_N)));        
        

