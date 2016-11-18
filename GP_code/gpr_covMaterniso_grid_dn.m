function [nlml, dnlml, alpha_kron] = gpr_covMaterniso_grid_dn(logtheta, xgrid, y, noise)
% tic
cov = {'covMaterniso', 5};
logtheta;
N = prod(cellfun(@length, xgrid));
D = length(xgrid);
if size(y,1) ~= N
    error('Invalid vector of targets, quitting...');
end
if length(logtheta) ~= D+1
    error('Error: Number of parameters do not agree with covariance function!')
end

Ks = cell(D,1);
Qs = cell(D,1);
QTs = cell(D,1);
Vs=cell(D,1);
for d = 1:D
    
    xg = xgrid{d};
    if size(xg,2) > size(xg,1)
        xg = xg';
    end
    hyp.cov = [logtheta(d); logtheta(D+1)/D];
    K_kron = feval(cov{:},hyp.cov, xg); %TODO: Toeplitz
    Ks{d} = K_kron;
    [Q,V] = eig(K_kron); %TODO: Toeplitz
    Qs{d} = Q;
    QTs{d} = Q';
    Vs{d}=V;
    V = diag(V);
    if d == 1
        V_kron = V;
    else
        V_kron = kron(V_kron, V); %this is a vector so still linear in memory
    end
    
end

V_kron = V_kron + 1e-10; %epsilon for computational stability
% V_kron_noise = V_kron + noise;  
%fprintf('Inverse condition number = %e\n', min(V_kron)/max(V_kron));

invC = ones(size((V_kron)));

% %%%
% temp = kron(diag(Ks{1}),diag(Ks{2}))+noise;
% invC = temp.^(-1).*ones(size((V_kron)));
% alpha_kron = pre_conj_grad_solve(Qs, V_kron, noise, y,invC);
% %%%

% temp = kron(diag(Ks{1}),diag(Ks{2}));         % only C=Kii
% invC = temp.^(-0.5).*ones(size((V_kron)));


%%%
invC = noise.^(-0.5).*ones(size((V_kron)));     %C = inv(noise)
% invC2 = noise.^(-0.6).*ones(size((V_kron)));     %C = inv(noise)
% invC3 = noise.^(-1).*ones(size((V_kron)));     %C = inv(noise)
% invC4 = noise.^(-0).*ones(size((V_kron)));     %C = inv(noise)

% dumIndx = noise<1e14;                         %C = inv(dummy noise only)
% invC = noise.^(-1).*ones(size((V_kron)));
% invC(dumIndx)=1;

% temp = Ks{1}.*Ks{2};
% invC = (temp(:)+noise).^(-1).*ones(size((V_kron)));     %C = inv(diag(K)+noise)

% tic
% alpha_kron1 = conj_grad_solve(Qs, V_kron, noise, y);
% toc
tic
max_iter = max(min(N,10000),1000);
[alpha_kron rs] = pre_conj_grad_solve(Qs, V_kron, noise, y,invC,max_iter);
fprintf('Conjugate gradient solver converged in %i iterations. logtheta =[%d,%d,%d]\n',length(rs),logtheta(1),logtheta(2),logtheta(3));
toc

% sum((alpha_kron1 - alpha_kron).^2)
if(length(rs) == max_iter)
    nlml = inf;
    dnlml = inf*ones(size(logtheta));
    return;
end
% alpha_kron = (kron(Ks{1},Ks{2})+diag(noise))\y;     %%naive alpha calculation

% sum((alpha_kron1 - alpha_kron).^2)

% figure;imagesc(reshape(real(alpha_kron),40,40))
%%%


% alpha_kron1 = kron_mv(QTs, y(1:N));
% alpha_kron1 = alpha_kron1./V_kron_noise;
% alpha_kron1 = kron_mv(Qs, alpha_kron1);
% sum((alpha_kron1 - alpha_kron).^2)
% alpha_kron = alpha_kron1;



% alpha_kron = conj_grad_solve(Qs, V_kron.^(-1), noise.^(-1), alpha_kron);
% alpha_kron = pre_conj_grad_solve(Qs, V_kron.^(-1), noise.^(-1), alpha_kron,invC);

% alpha_kron = kron_mv(QTs, alpha_kron);
% alpha_kron = alpha_kron./V_kron;
% 
% 
% 
% alpha_kron2 = kron_mv(QTs, y(1:N));     % Need to optimize this part
% alpha_kron2 = alpha_kron2./V_kron;

% alpha_kron = kron_mv(Qs, alpha_kron+alpha_kron2);


% %%%%%%
% alpha_kron = kron_mv(Qs, y(1:N));
% alpha_kron = alpha_kron./sqrt(V_kron);
% % invC = ones(size((V_kron))); %((V_kron.^(1/2)).*noise).^(-1).*ones(size((V_kron)));
% invC = kron_mv(QTs, kron_mv(Qs, ones(size(y)))./((V_kron.^(-1)).*noise)).^(-1);
% alpha_kron = pre_conj_grad_solve(QTs, (V_kron.^(-1)).*noise, ones(size(noise)), alpha_kron,invC);
% %%%%%%


% toc
% figure(31); subplot(10,2,15); imagesc(reshape(V_kron,40,40));colorbar;xlabel([num2str(logtheta')]); subplot(10,2,16); imagesc(reshape(alpha_kron,40,40)); xlabel('106')

G1 = size(Ks{1},1);
G2 = size(Ks{2},1);

% K1inv = (Qs{1}*diag(diag(Vs{1}+1e-15*eye(size(Vs{1},1))).^(-1))*QTs{1});
% K2inv = real(Qs{2}*diag(diag(Vs{2}+1e-15*eye(size(Vs{1},1))).^(-1))*QTs{2});

% noiseinv = noise.^(-1);
% logdet_kron=sum(log(real(V_kron)))+sum(log(noise));
% for i = 1:G1
%     d = eig(K2inv+diag(noiseinv((i-1)*G2+1:i*G2)));
%     logdet_kron=logdet_kron+G2*log(K1inv(i,i))+sum(log(real(d)));
% end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%% diagonal estimation first order
% logdet_kron2=0;
% K1 = Ks{1};
% K2=Ks{2};
% for i = 1:G1
%     d = eig(K2+diag( noise( (i-1)*G2+1:i*G2 ) ));
%     logdet_kron2=logdet_kron2+G2*log(K1(i,i))+sum(log(real(d)));
% end

% %%%%%%%%%%%%%%%%%%%%%%%%%%% other diagonal estimatino, doesn't work
% logdet_kron2=0;
% K1 = Ks{1};
% K2=Ks{2};
% for i = 1:G1
%     d = eig(K2+diag( noise( (i-1)*G2+1:i*G2 ) ));
%     logdet_kron2=logdet_kron2+G2*log(K1(i,i))+sum(log(real(d)));
% end
% logdet_kron2=logdet_kron2+G2*sum(log(abs(eig(K1-diag(diag(K1))))));

% %%%%%%%%%%%%%%%%% geometrical mean with dummy noise
gavg_noise = exp(sum(log(noise))/length(noise));
logdet_kron = sum(log(V_kron+gavg_noise));
logdet_kron2 = logdet_kron;

% % %%%%%%%%%%%%%%%%% geometrical mean without dummy noise
% noise1 = noise(noise < 1e14);
% gavg_noise = exp(sum(log(noise1))/length(noise1));
% logdet_kron = sum(log(V_kron+gavg_noise));
% logdet_kron2 = logdet_kron;


% logdet_kron = sum(log( eig(kron(Ks{1},Ks{2})+diag(noise))));      %%naive logdet calculation



% logdet_kron2 = sum(log( eig(kron(Ks{1},Ks{2}))))+...
%     sum(log(1+noise.*kron(diag(K1inv),diag(K2inv))));


% Ky = kron(Ks{1},Ks{2})+diag(noise);
% Ky = Ky(noise<1e14,noise<1e14);
% logdet_kron = sum(log( eig(Ky)));      %%naive logdet calculation without dummy

% logdet_kron = sum(log(V_kron_noise))

% figure;imagesc(Ks{1})
% disp([num2str(alpha_kron'*y),' ',num2str(real(alpha_kron'*y)),' ',num2str(logdet_kron),' ',num2str(real(logdet_kron))]);


figure(10);imagesc((reshape(real(alpha_kron.*y),G2,G1)))
figure(11);imagesc((reshape(real(alpha_kron),G2,G1)))

nlml = real(0.5*((alpha_kron')*y + (logdet_kron) + N*log(2*pi)));

disp([num2str(alpha_kron'*y),' ',num2str(logdet_kron),' ',num2str(real(nlml))]);
if(nlml == -inf)
    nlml = inf;
    dnlml = inf*ones(size(logtheta));
    return
end


%Now for the derivatives
dnlml = zeros(size(logtheta));

% %lengthscales
% ell = exp(logtheta(1:D));                         
% for i = 1:D
%     dC = cell(D,1);
%     dK = cell(D,1);
%     for d = 1:D
%         if d == i
%             dK_kron = Ks{d} .* sq_dist(xgrid{i}/ell(i));
%         else
%             dK_kron = Ks{d};
%         end
%         dK{d} = dK_kron;
%         dC{d} = QTs{d}*dK_kron'*Qs{d};
%         if d == 1
%             diag_Z = diag(dC{d});
%         else
%             diag_Z = kron(diag_Z, diag(dC{d}));
%         end
%     end
%     dnlml(i) = 0.5*(sum(diag_Z./V_kron_noise) - alpha_kron'*kron_mv(dK, alpha_kron));
% end

%lengthscales                         
for i = 1:D
    dC = cell(D,1);
    dK = cell(D,1);
    for d = 1:D
        xg = xgrid{d};
        if size(xg,2) > size(xg,1)
            xg = xg';
        end
        if d == i
            dK_kron = feval(cov{:},hyp.cov, xg,[],1); %1 = ell
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
    dnlml(i) = 0.5*(sum(diag_Z./(V_kron+gavg_noise)) - alpha_kron'*kron_mv(dK, alpha_kron));
%     dnlml(i) = 0.5*(sum(diag_Z./(V_kron+mean(noise))) - alpha_kron'*kron_mv(dK, alpha_kron));
end


% 
% %amplitude & noise
% dC = cell(D,1);
% dN = cell(D,1);
% for d = 1:D
%     dC{d} = QTs{d}*Ks{d}'*Qs{d};
%     dN{d} = QTs{d}*Qs{d};
%     if d == 1
%         diag_Z = diag(dC{d});
%         diag_N = diag(dN{d});
%     else
%         diag_Z = kron(diag_Z, diag(dC{d}));
%         diag_N = kron(diag_N, diag(dN{d}));
%     end
% end
% dnlml(D+1) = sum(diag_Z./V_kron_noise) - alpha_kron'*kron_mv(Ks, alpha_kron);
% dnlml = real(dnlml);


% %amplitude
dC = cell(D,1);
dK = cell(D,1);
for d = 1:D
    xg = xgrid{d};
    if size(xg,2) > size(xg,1)
        xg = xg';
    end
%     if d == i
        dK_kron = feval(cov{:},hyp.cov, xg,[],2); %2 = sf2
%     else
%         dK_kron = Ks{d};
%     end
    dK{d} = dK_kron;
    dC{d} = QTs{d}*dK_kron'*Qs{d};
    if d == 1
        diag_Z = diag(dC{d});
    else
        diag_Z = kron(diag_Z, diag(dC{d}));
    end
end
dnlml(D+1) = 0.5*(sum(diag_Z./(V_kron+gavg_noise)) - alpha_kron'*kron_mv(dK, alpha_kron));
% dnlml(D+1) = 0.5*(sum(diag_Z./(V_kron+mean(noise))) - alpha_kron'*kron_mv(dK, alpha_kron));

dnlml = real(dnlml);
disp([nlml,dnlml']);

