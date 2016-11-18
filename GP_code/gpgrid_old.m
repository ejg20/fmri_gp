function [Xfull, logtheta_learned] = gpgrid_old(Isub,Xgrid,xsize, ysize, suglogtheta, learnflag)

% [xsize,ysize] = size(Isub);

% [X1,X2] = meshgrid(1:upsamp:upsamp*xsize,1:upsamp:ysize);

if(nargout > 1) 
    prediction = false;
else
    prediction = true;
end
y = Isub(:);
D = length(Xgrid);

if(nargin<5)
    suglogthetaFlag=false;
    s0 = var(y)+1e-5;
    sf =ones(D,1)*sqrt(s0/4); 
    sn = sqrt(s0)/100;
else
     suglogthetaFlag=true;
     if(suglogtheta(1) == inf)
         suglogthetaFlag=false;
     end
     sf =suglogtheta(D+1); 
     sn = suglogtheta(D+2);
end

if(nargin <6)
    learnflag = true;
end

% x = [X1(:) X2(:)];

% Y = Isub;

%[N,D] = size(x);

[xstarvec,ystarvec] = meshgrid(1:1:xsize,1:1:ysize);
% xstarvec = xstarvec';
% ystarvec = ystarvec';
xstar = [xstarvec(:),ystarvec(:)];

% cov_func = 'gpr_covMaterniso_grid';

if(suglogthetaFlag == false)
    % ellscoef = [1 5 10 20 50 100];
    ellscoef = 2.^(0:10);
    parfor ellsi = 1:length(ellscoef)
        ells = ellscoef(ellsi)*ones(1,D);
        
    %     logtheta_init = log([ells'; ones(D,1)*sqrt(s0); 0]);
        logtheta_init = log([ells'; sf; sn]); 

        % [xsize, ysize] = size(Xfull);

        % M = size(xstar,1);


        %%% Kronecker

        % Xgrid1 = cell(D,1);
        % for d = 1:D
        %     Xgrid1{d} = 1:2:xsize;
        % end

        % Xgrid{3} = 10;
        % likfunc = @likGauss;
        % covfunc = @gpr_covSEard_grid; hyp.cov = log(2.2*rand(D,1)); hyp.lik = log(0.1);
        % %disp('hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfunc, likfunc, x, y)');
        % hyp = minimize(hyp, @gp, -100, @infExact, [], covfunc, likfunc, x, y);
        % disp(' ');

        % logtheta = log([2.2*rand(D,1); 1;0.1]);
        %tic
    %     [logtheta_learned fx] = minimize([logtheta_init(1:D); logtheta_init(D+1); logtheta_init(end)], 'gpr_covSEard_grid', 100, Xgrid, y);
         [logtheta_learned fx] = minimize([logtheta_init(1:D); logtheta_init(D+1); logtheta_init(end)], 'gpr_covMaterniso_grid', 100, Xgrid, y);

        fxt(ellsi) = fx(end);
        logtheta_learnedt(:,ellsi) = logtheta_learned;
        %  
    end
    [fmin imin] = min(fxt);
    logtheta_learned = logtheta_learnedt(:,imin);
else
    if(learnflag == true)
        [logtheta_learned fx] = minimize(suglogtheta, 'gpr_covMaterniso_grid', 100, Xgrid, y);
%         [logtheta_learned fx] = minimize(suglogtheta, 'gpr_covMaterniso_grid', 100, Xgrid, y);
    else
        logtheta_learned = suglogtheta;
    end
end

% [nlml, dnlml, alpha_kron] = gpr_covSEard_grid(logtheta_learned, Xgrid, y);
[nlml, dnlml, alpha_kron] = gpr_covMaterniso_grid(logtheta_learned, Xgrid, y);
%learnexectime =toc
tic
% [i,j] = ind2sub([xsize,ysize],XtestInd)
% [i,j] = ind2sub([xsize,ysize],1:prod([xsize,ysize]));


% Xfull = zeros(xsize,ysize);
% if(prediction)
%     parfor i = 1:xsize*ysize
%         [f, df] = gpr_covMaterniso_grid_predict(xstar(i,:)', logtheta_learned, Xgrid, y, alpha_kron);
%         Xfull(i) = f;
%     end
%     predictexectime = toc
%     figure(6); imagesc(Xfull); drawnow
% end
Xfull = zeros(ysize,xsize);
if(prediction)
    [mu_f, std_f] = gpr_covMaterniso_grid_predict_multi(xstar, logtheta_learned, Xgrid, y, alpha_kron);
    Xfull(sub2ind(size(Xfull),xstar(:,2),xstar(:,1))) = mu_f;
    figure(7); imagesc(Xfull); drawnow
end

% for m = 1:size(xstar,1)
%     Xfull(xstar(m,1),xstar(m,2)) = mu_f(m);
% end
% figure(8); imagesc(Xfull); drawnow
% save('gpr_grid_results')
% close all;
% fig1 = figure;
% axes1 = axes('Parent',fig1);
% view(axes1,[-26 34]);
% hold(axes1,'all');
% h = surface(xstarvec(1,:),ystarvec(:,1),real(fbar))
% axis([1 40 1 40 1, 200])
% Fmovie(iframe) = getframe;

end
