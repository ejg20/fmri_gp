function [Xfull, logtheta_learned,Stdfull] = gpgrid_img_dn(Isub,Xgrid, ver_size, hor_size, noise, mask)

    cov = {'covMaterniso', 5};

% [xsize,ysize] = size(Isub);

% [X1,X2] = meshgrid(1:upsamp:upsamp*xsize,1:upsamp:ysize);
Xfull = [];
% if(nargout > 1)
%     prediction = false;
% else
    prediction = true;
% end
y = Isub(:);
D = length(Xgrid);


s0 = var(y)+1e-5;
sf =ones(D,1)*sqrt(s0)/10;



% cov_func = 'gpr_covMaterniso_grid';


% ellscoef = [1 5 10 20 50 100];
ellscoef = 2.^(1:3:7);
sfcoef = 2.^[1,5];
% ellscoef = 2.^5;
for sfi = 1:length(sfcoef)
    parfor ellsi = 1:length(ellscoef)
        ells = ellscoef(ellsi)*ones(D,1);

        %     logtheta_init = log([ells'; ones(D,1)*sqrt(s0); 0]);
        logtheta_init = log([ells; sfcoef(sfi)]);


%             [logtheta_learned fx] = minimize([logtheta_init(1:D); logtheta_init(D+1); logtheta_init(end)], 'gpr_covSEard_grid', 100, Xgrid, y);
%         [logtheta_learned fx] = minimize([logtheta_init(1:D); logtheta_init(D+1)], 'gpr_covMaterniso_grid_dn', -40, Xgrid, y, noise);
        [logtheta_learned fx] = minimize(logtheta_init, 'gpr_cov_grid_dn', -40, Xgrid, y, noise, cov);
%         
%         logtheta_learned = logtheta_init;
%         [fx, dnlml, alpha_kron] = gpr_covMaterniso_grid_dn(logtheta_learned, Xgrid, y, noise);

        fxt(ellsi) = fx(end);
        logtheta_learnedt(:,ellsi) = logtheta_learned;
        %
    end
    [fmin imin] = min(fxt);
    logtheta_learnedtt(:,sfi) = logtheta_learnedt(:,imin);
    fmin_learnedtt(sfi) = fmin;
    imin_learnedtt(sfi) = imin;
end
[fmin imin] = min(fmin_learnedtt);
logtheta_learned = logtheta_learnedtt(:,imin);

% disp(['sf learned = 2^',num2str(imin-1),' ells learned = 2^',num2str(imin_learnedtt(imin)-1)]);
(logtheta_learned)./log(2)
exp(logtheta_learned)
% [nlml, dnlml, alpha_kron] = gpr_covSEard_grid(logtheta_learned, Xgrid, y);
% [nlml, dnlml, alpha_kron] = gpr_covMaterniso_grid_dn(logtheta_learned, Xgrid, y, noise);
[nlml, dnlml, alpha_kron, Qs, V_kron] = gpr_cov_grid_dn(logtheta_learned, Xgrid, y, noise, cov);
%learnexectime =toc

% keyboard; 
% return;
% tic

% tic
if(prediction)
    [hor_starvec,ver_starvec] = meshgrid(1:1:ver_size,1:1:hor_size);
    xstar = [ver_starvec(:),hor_starvec(:)];
    xstar(logical(mask(sub2ind(size(mask),xstar(:,2),xstar(:,1)))),:) = [];
Xfull = zeros(ver_size,hor_size);
Stdfull = zeros(ver_size,hor_size);
%     [mu_f, std_f] = gpr_covMaterniso_grid_predict_multi(xstar, logtheta_learned, Xgrid, y, alpha_kron);   
    [mu_f, var_f] = gpr_cov_grid_predict_parallel(xstar, logtheta_learned, Xgrid, y, alpha_kron, noise, cov, Qs, V_kron);
    Xfull(sub2ind(size(Xfull),xstar(:,2),xstar(:,1))) = mu_f;
    Stdfull(sub2ind(size(Xfull),xstar(:,2),xstar(:,1))) = var_f;
    figure(7); imagesc(Xfull); drawnow
    figure(8); imagesc(log(Stdfull)); drawnow
end
% toc
% tic
% if(prediction)
%     [mu_f, std_f] = gpr_covMaterniso_grid_predict_multi_old(xstar, logtheta_learned, Xgrid, y, alpha_kron);
%     Xfull(sub2ind(size(Xfull),xstar(:,2),xstar(:,1))) = mu_f;
%     Stdfull(sub2ind(size(Xfull),xstar(:,2),xstar(:,1))) = std_f;
%     figure(7); imagesc(Xfull); drawnow
% end
% toc


end
