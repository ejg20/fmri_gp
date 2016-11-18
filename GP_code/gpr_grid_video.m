% clear;
close all;
clear variables
subSampling = 1;
load('Smoke.mat');

zgrid = [1:1:30,34,35];
vid = vid(41:140,541:640,zgrid);


% p = [0.2495   15.9858];
% p = [0   15.9858];
p = [0 1];

esn = (p(1)*(vid)+p(2));
sn = zeros(size(vid));
sn((vid == 0)) =  300*max(vid(:));
sn = sn+esn;

mask = (vid ~= 0);
% sn = sn+esn;
[xsize,ysize,zsize] = size(vid);
xsubindex = 1:xsize;
ysubindex = 1:ysize;
zsubindex = zgrid;

Xgrid = cell(3,1);
Xgrid{3} = xsubindex;
Xgrid{2} = ysubindex;
Xgrid{1} = zsubindex;
clear initHypGuess;
initHypGuess.l = 2.^(1:3:10);
initHypGuess.sf = 2.^[1,5,8];
initHypGuess.sn = 1;
% cov = {'covMaterniso', 5};
cov = {'covMaterniso', 3};
% cov = {'covSEard'};
%

[x_starvec,y_starvec,z_starvec] = meshgrid(1:xsize,1:ysize,31:33);
xstar = [z_starvec(:),y_starvec(:),x_starvec(:)];
% remove locations outside the mask
% xstar(logical(~mask(sub2ind(size(mask),xstar(:,3),xstar(:,2),xstar(:,1)))),:) = [];
% xstar = xstar(:,1:D);

input.index_to_N = find(vid > 0);
input.data = vid(input.index_to_N);
meanData = mean(input.data );
input.data = input.data-meanData;
%  [logtheta_learned, Iout] = gpgrid_dn(vid, Xgrid, [xsize, ysize, zsize], sn(:), xstar, initHypGuess, cov);
[logtheta_learned, Iout] = gpgrid_dn(input, Xgrid, [xsize, ysize, zsize], sn(:), xstar, initHypGuess, cov);

% return
%%
close all
figure;

colorlimit = [min(min(min(vid)))*0+30 max(max(max(vid)))]; colormap(gray)
subplot(3,3,1); imagesc(vid(:,:,end-4)); caxis manual; caxis(colorlimit);  
set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', [],'YTickLabelMode', 'manual', 'YTickLabel', [],'FontSize',16); xlabel('t = 49');
subplot(3,3,2); imagesc(vid(:,:,end-3)); caxis manual; caxis(colorlimit); 
set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', [],'YTickLabelMode', 'manual', 'YTickLabel', [],'FontSize',16); xlabel('t = 50');
subplot(3,3,3); imagesc(vid(:,:,end-2)); caxis manual; caxis(colorlimit); 
set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', [],'YTickLabelMode', 'manual', 'YTickLabel', [],'FontSize',16); xlabel('t = 51');
a = reshape(Iout,ysize,xsize,3)+meanData;
subplot(3,3,4); imagesc(a(:,:,1)'); caxis manual; caxis(colorlimit); 
set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', [],'YTickLabelMode', 'manual', 'YTickLabel', [],'FontSize',16); xlabel('t = 52');
subplot(3,3,5); imagesc(a(:,:,2)'); caxis manual; caxis(colorlimit); 
set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', [],'YTickLabelMode', 'manual', 'YTickLabel', [],'FontSize',16); xlabel('t = 53');
subplot(3,3,6); imagesc(a(:,:,3)'); caxis manual; caxis(colorlimit);
set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', [],'YTickLabelMode', 'manual', 'YTickLabel', [],'FontSize',16); xlabel('t = 54');

subplot(3,3,7); imagesc(vid(:,:,end-1)); caxis manual; caxis(colorlimit); 
set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', [],'YTickLabelMode', 'manual', 'YTickLabel', [],'FontSize',16); xlabel('t = 55');
subplot(3,3,8); imagesc(vid(:,:,end)); caxis manual; caxis(colorlimit); 
set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', [],'YTickLabelMode', 'manual', 'YTickLabel', [],'FontSize',16); xlabel('t = 56');
% subplot(3,3,9); imagesc(vid(:,:,54)); caxis manual; caxis(colorlimit); 
% set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', [],'YTickLabelMode', 'manual', 'YTickLabel', [],'FontSize',16); xlabel('t = 57');


% [i,j] = ind2sub([xsize,ysize],XtestInd)
    % [i,j] = ind2sub([xsize,ysize],1:prod([xsize,ysize]));
    
    % fbar = zeros(xsize,ysize);
    % parfor i = 1:xsize*ysize
    %     [f, df] = gpr_covSEard_grid_predict(xstar(i,:)', logtheta_learned, Xgrid, y, alpha_kron);
    %     fbar(i) = f;
    %     i
    % end
    % predictexectime = toc
    % % save('gpr_grid_results')
    % close all;
    % fig1 = figure;
    % axes1 = axes('Parent',fig1);
    % view(axes1,[-26 34]);
    % hold(axes1,'all');
    % h = surface(xstarvec(1,:),ystarvec(:,1),real(fbar))
    % axis([1 40 1 40 1, 200])
    % Fmovie(iframe) = getframe;
    % end
    return;
    
    
    
    % for i=1:ceil(size(xstar,1)/100)
    %     startIdx = (i-1)*100+1;
    %     endIdx = startIdx + min(100,size(xstar,1)-startIdx) - 1;
    %     [f, df] = gpr_covSEard_grid_predict(xstar(startIdx:endIdx,:), logtheta_learned, Xgrid, y, [], [],locate_info,0);
    %     if(length(f)<100)
    %         f(100)=0;
    %     end
    %     fbar(:,i) = f;
    % end
    %
    % fbar = f;
    % vid(xstar(1),xstar(2),1)
    
    
    
    
    %% Full GP (tensor)
    
    
    tic;
    
    %        y((ysize/2)*(xsize/4)+(xsize/4)) = 30;
    
    
    
    
    covfunc = @covSEiso;
    likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
    hyp2.cov = [0 ; 0];
    hyp2.lik = log(0.1);
    
    hyp2 = minimize(hyp2, @gp, -50, @infExact, [], covfunc, likfunc, x, y);
    %%
    [mus vars] = gp(hyp2, @infExact, [], covfunc, likfunc, x, y, xstar);
    
    %%
    %     figure;
    %     imshow(reshape(mus,size(Y)),[0,255])
    
    figure
    z = (1:xsize)';
    mus1 = reshape(mus,size(xstarvec));
    vars1 = reshape(vars,size(xstarvec));
    mus1 = mus1((xsize/2),:)';
    vars1 = vars1((xsize/2),:)';
    y1 = reshape(y,size(X1));
    f = [mus1 + sqrt(vars1);flipdim(mus1 - sqrt(vars1),1)];
    fill([z; flipdim(z,1)], f, [7 7 7]/8)
    hold on
    plot(1:2:xsize,y1((xsize/2),:),'--r')
    hold on
    plot(1:xsize,mus1)
    %     plot(1:20,vid(1:20,1,3),'--g')
    hold off
    legend('confidence','frame(2)','GP interpolation','frame(3)')
    
    nlml2 = arrayfun(@(i)gp(hyp2, @infExact, [], covfunc, likfunc, x(i,:), y(i)),(1:length(y)));
    nlml2 = reshape(nlml2, size(Y));
    
    figure
    surface(xstarvec(1,:),ystarvec(:,1),reshape(mus,size(xstarvec)))
    figure
    surface(xstarvec(1,:),ystarvec(:,1),reshape(sqrt(vars),size(xstarvec)))
    
    figure
    surface(xstarvec(1,:),ystarvec(:,1),reshape(mus,size(xstarvec)),zeros(size(xstarvec)))
    hold on
    surface(X1(1,:),X2(:,1),Y,nlml2)
    hold off
    
    
    figure
    surface(X1(1,:),X2(:,1),reshape(nlml2,size(X1)))
    
    return
    mnlp_test = 0.5*(log(2*pi) + mean(log(vars)) + mean(((ystar - mus).^2)./vars));
    nmse_test = mean(((ystar - mus).^2))/mean(ystar.^2);
    fprintf('Testing NMSE (joint GP) = %3.5f\n', nmse_test);
    fprintf('Testing MNLP (joint GP) = %3.5f\n', mnlp_test);
    exec_time = toc;
    fprintf('Execution time (joint GP) = %5.1f\n', exec_time);
    
    full_gp.mnlp = mnlp_test;
    full_gp.nmse = nmse_test;
    full_gp.logtheta = logtheta;
    full_gp.exec_time = exec_time;
    
    %end
