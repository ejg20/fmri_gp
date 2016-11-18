clear variables
close all


global prodcovfunc alpha_prev;
prodcovfunc = 'gp_grid_prodcov_v5';  
alpha_prev = 0;

%% create interesting kernels
n=50;
fntsz = 14;
figure(1)
x = (1:n)';
D=1;
covfunc = {@covMaterniso, 3}; ell = 20; sf = 1; hyp.cov = log([ell; sf]);
SE1 = feval(covfunc{:}, hyp.cov, x);

covfunc = {@covSEard}; ell = 5; sf = 1; hyp.cov = log([ell; sf]);
SE2 = feval(covfunc{:}, hyp.cov, x);

covfunc = {@covPeriodic}; ell = 1; p = 10; sf = 3; hyp.cov = log([ell; p; sf]);
PER1 = feval(covfunc{:}, hyp.cov, x);

K1 = SE1;
% K1 = (SE1+SE2).*PER1;
%  K1 = SE1;%+PER1;
K10 = K1(1);

K1 = K1/K10; K10=1;
subplot(1,3,1); plot(0:size(K1,2)-1,K1(1,:)/K10,'g'); xlabel('\tau', 'fontsize',fntsz+2); ylabel('$k_1$','Interpreter','Latex', 'fontsize',fntsz+5)
set(gca, 'fontsize',fntsz);

covfunc = {@covMaterniso, 1}; ell = 20; sf = 1; hyp.cov = log([ell; sf]);
MA1 = feval(covfunc{:}, hyp.cov, x);

covfunc = {@covPeriodic}; ell = 2; p = 5; sf = 1; hyp.cov = log([ell; p; sf]);
PER1 = feval(covfunc{:}, hyp.cov, x);

covfunc = {@covMaterniso, 1}; ell = 10; sf = 1; hyp.cov = log([ell; sf]);
MA2 = feval(covfunc{:}, hyp.cov, x);

covfunc = {@covPeriodic}; ell = 15; p = 15; sf = 1; hyp.cov = log([ell; p; sf]);
PER2 = feval(covfunc{:}, hyp.cov, x);

K2 = MA1.*(PER1+MA2).*PER2;
% K2 = MA1;
K20 = K2(1);
K2 = K2/K20; K20=1;
subplot(1,3,2); plot(0:size(K2,2)-1,K2(1,:)/K20,'g'); xlabel('\tau', 'fontsize',fntsz+2); ylabel('$k_2$','Interpreter','Latex', 'fontsize',fntsz+5)
set(gca, 'fontsize',fntsz);

covfunc = {@covRQard}; ell = 30; sf = 1; alpha = 2; hyp.cov = log([ell; sf; alpha]);
RQ1 = feval(covfunc{:}, hyp.cov, x);

covfunc = {@covPeriodic}; ell = 1; p = 10; sf = 1; hyp.cov = log([ell; p; sf]);
PER1 = feval(covfunc{:}, hyp.cov, x);

covfunc = {@covMaterniso, 3}; ell = 10; sf = 1; hyp.cov = log([ell; sf]);
SE1 = feval(covfunc{:}, hyp.cov, x);

covfunc = {@covPeriodic}; ell = 5; p = 8; sf = 1; hyp.cov = log([ell; p; sf]);
PER2 = feval(covfunc{:}, hyp.cov, x);

K3 = (RQ1+PER1+SE1);%.*PER2+SE1;
% K3 = (RQ1+SE1);
K30 = K3(1);
K3 = K3/K30; K30=1;
subplot(1,3,3); plot(0:size(K3,2)-1,K3(1,:)/K30,'g');  xlabel('\tau', 'fontsize',fntsz+2); ylabel('$k_3$','Interpreter','Latex', 'fontsize',fntsz+5)

% subplot(1,3,3); plot(0:size(K1,2)-1,K1(1,:)+K2(1,:),'g');  xlabel('\tau', 'fontsize',fntsz+2); ylabel('$k_3$','Interpreter','Latex', 'fontsize',fntsz+5)


set(gca, 'fontsize',fntsz);
%% generate data

% product kernels test
% Ks = {(K1),(K2),(K3)};
% cholKs = {chol(K1),chol(K2),chol(K3)};
% v = gpml_randn(0.52276, size(cholKs,1), 1);
% yvec = kron_mv(cholKs,v);
% y = reshape(yvec,[n,n,n]);
Ks1 = {(K1),(K2)};
cholKs1 = {chol(K1)',chol(K2)'};
Ks2 = {(K3),(K3)};
cholKs2 = {chol(K3)',chol(K3)'};
v = gpml_randn(0.52276, n^2, 1);
% yvec = kron_mv(cholKs1,v)+kron_mv(cholKs2,v);
Kbig = kron(K1,K2)+kron(K3,K3);
yvec = (chol(Kbig)')*v;
y = reshape(yvec,[n,n]);

% sum test


% cholKs = chol(K1+K2);
% cholKs = chol(K1)';
% v = gpml_randn(0.20285, size(cholKs,1), 1);
% yvec = cholKs*v;
% y = yvec;

% bestfit = inf;
% for i = 1:100
%     a = rand; v = gpml_randn(a, n, 1);
%     dfit = v'*(K1\v)
%     if(dfit < bestfit)
%         bestfit = dfit;
%         bestseed = a;
%     end
% end

% check for good seed for maximizing probability
% a = rand; v = gpml_randn(a, n^3, 1);v'*kron_mv(invKs,v)

% midn = fix(n/2);
% testinx = [midn-2,midn-1,midn,midn+1,midn+2];
% figure(2); 
% for i = 1:length(testinx)
% subplot(2,5,i); imagesc(y(:,:,testinx(i)));
% colormap(gray)
% axis off
% caxis([-30       63.932])
% end



%% run gp_grid to recover kernel
ytr = y; 


figure(1)
subplot(1,2,1); CAX = showPatternPlot(ytr,[],'Original Pattern');
freezeColors;
cbfreeze(colorbar);
subplot(1,2,2); showPatternSpectrumPlot(ytr,[],'Log magnitude spectrum',1);
drawnow;

%ytr(:,:,testinx)=[];

gp_input = gp_grid_input_class(ytr,(1:numel(ytr)),1);

%have to change the grid manually for now
% gp_input.xgrid{1} = [1:testinx(1)-1,testinx(end)+1:n]'; %xgrid dim are flipped

noise_struct = gp_grid_noise_class(ones(size(ytr)),gp_input.index_to_N);
noise_struct.learn = true;

covs{1} = {{'covSM1D',1},{'covSM1D',1}};
covs{2} = {{'covSM1D',20},{'covSM1D',20}};
gpmodel = gp_grid_gpmodel_class();
gpmodel.cov = covs;
gpmodel.noise_struct = noise_struct;
gpmodel.hyps_in_d  = make_hyps_in_d_v2(covs);
gpmodel.learn = true;

Z = length(gpmodel.cov);
P = length(gpmodel.cov{1});

lambda = 1e4;
gpmodel.logpriorfuns{1}.func = @(t) gp_grid_Laplace(t, lambda);

runingindex = 0;
windx=[];
for z = 1:Z
    for p = 1:P
        Q = gpmodel.cov{z}{p}{2};
        windx = [windx;runingindex+(1:Q)'];
        runingindex = runingindex +  3*Q;
    end
end
gpmodel.logpriorfuns{1}.indices = windx;

% [r c]  = (ind2sub(size(mask),find(mask < 1)));
% xstar = [c(:) r(:)];
% xstar = [1,1,1];
xstar = [];
params.wm = std(gp_input.zeromeandata);
Gs = gp_grid_size(gp_input.xgrid);
params.sm = 2*Gs./gp_input.Fs;

% besttrnlml = inf;
% for i = 1:4
%     gpmodel.learn = true;
%     [hypers_learned, trnlml, Iout_z, Iout, Vout, filename] = run_gp_grid_v2(gp_input, gpmodel,'params',params,'xstar',xstar,...
%         'lognoise',-5,'numofstarts',20,'maxiteration',50,'filename',['bigdata',num2str(covs{1}{1}{2})]);
%     if(besttrnlml > trnlml.nlml)
%         besttrnlml = trnlml.nlml;
%         bestinit = hypers_learned;
%     end
% end


gpmodel.learn = true;

[hypers_learned, trnlml, Iout_z, Iout, Vout, filename] = run_gp_grid_v2(gp_input, gpmodel,'params',params,'xstar',xstar,...
    'lognoise',-5,'numofstarts',20,'maxiteration',200,'filename',['bigdata',num2str(covs{1}{1}{2})]);
bestinit = hypers_learned;

% 
% [hypers_learned, trnlml, Iout_z, Iout, Vout, filename] = run_gp_grid_v2(gp_input, gpmodel,'params',params,'xstar',xstar,...
%     'lognoise',-5,'numofstarts',50,'maxiteration',100,'filename',['bigdata',num2str(covs{1}{1}{2})],'hypers_init',bestinit);
% 


% pattLearned = y;
% pattLearned(mask == 0) = Iout;
% figure; imagesc(pattLearned)



%% spectrum

% specdensityImage = multdimSpecdensity({0:0.001:0.5},hypers_learned,gpmodel.hyps_in_d{1});
% figure(4); plot(0:0.001:0.5,log(abs(specdensityImage)));
% hold on
% plot(log(abs(fft(K1(1,:),1000))))
% hold off

figure(4); 
subplot(1,3,1); CAX = showPatternPlot(y,[],'Original Pattern');
freezeColors;
cbfreeze(colorbar);
figure; showPatternSpectrumPlot(y,[],'Log magnitude spectrum',gp_input.Fs);
drawnow;
figure; showSMkernel_v2(hypers_learned, gpmodel.hyps_in_d, gp_input.Fs)


%% plot recovered kernel
linsz=3;
figure(1)
subplot(1,3,1); plot(0:size(K1,2)-1,K1(1,:)/K10,'g','LineWidth',linsz);  xlabel('\tau', 'fontsize',fntsz+2); ylabel('$k_1$','Interpreter','Latex', 'fontsize',fntsz+5)
set(gca, 'fontsize',fntsz);
subplot(1,3,2); plot(0:size(K2,2)-1,K2(1,:)/K20,'g','LineWidth',linsz);  xlabel('\tau', 'fontsize',fntsz+2); ylabel('$k_2$','Interpreter','Latex', 'fontsize',fntsz+5)
set(gca, 'fontsize',fntsz);
subplot(1,3,3); plot(0:size(K3,2)-1,K3(1,:)/K30,'g','LineWidth',linsz);  xlabel('\tau', 'fontsize',fntsz+2); ylabel('$k_3$','Interpreter','Latex', 'fontsize',fntsz+5)
set(gca, 'fontsize',fntsz);


% %%%% TAKE AWAY %%%%
%     gpmodeltemp = gp_grid_gpmodel_class(gpmodel);
%     hyd1 = gpmodel.hyps_in_d{1}([1:20,41:60,81:100]);
%     hyd2 = gpmodel.hyps_in_d{1}(20+[1:20,41:60,81:100]);
%     gpmodeltemp.hyps_in_d = {{hyd1},{hyd2}};
%     gpmodeltemp.cov{2} = gpmodel.cov{2}/2;
%     %%%%%%%%%%%%%%%%%%%%


covfunc = gpmodel.cov{1}; hyp.cov = hypers_learned(gpmodel.hyps_in_d{1}{1});
K1_rec = feval(covfunc{1}{:}, hyp.cov, x(1), x);
K1r0 = K1_rec(1);
subplot(1,3,1); 
hold on; plot(0:size(K1_rec,2)-1,K1_rec(1,:)/K1r0,'LineWidth',linsz); hold off
%xlabel('\tau','fontsize',fntsz+2); ylabel('$\hat{k}_1$','Interpreter','Latex', 'fontsize',fntsz+5)
set(gca, 'fontsize',fntsz);

covfunc = gpmodel.cov{1}; hyp.cov = hypers_learned(gpmodel.hyps_in_d{1}{2});
K2_rec = feval(covfunc{1}{:}, hyp.cov, x(1), x);
K2r0 = K2_rec(1);
subplot(1,3,2); 
hold on; plot(0:size(K2_rec,2)-1,K2_rec(1,:)/K2r0,'LineWidth',linsz); hold off
% xlabel('\tau','fontsize',fntsz+2); ylabel('$\hat{k}_2$','Interpreter','Latex', 'fontsize',fntsz+5)
set(gca, 'fontsize',fntsz);


covfunc = covs{2}; hyp.cov = hypers_learned(gpmodel.hyps_in_d{2}{1});
K3_rec = feval(covfunc{2}{:}, hyp.cov, x(1), x);
K3r0 = K3_rec(1);
subplot(1,3,3); 
hold on; plot(0:size(K3_rec,2)-1,+K3_rec(1,:)/(K3r0),'LineWidth',linsz); hold off;
xlabel('\tau','fontsize',fntsz+2); ylabel('$\hat{k}_3$','Interpreter','Latex', 'fontsize',fntsz+5)
set(gca, 'fontsize',fntsz);

covfunc = covs{2}; hyp.cov = hypers_learned(gpmodel.hyps_in_d{2}{2});
K4_rec = feval(covfunc{2}{:}, hyp.cov, x(1), x);

% K3_rec = K1_rec/K1r0+K2_rec/K2r0;
% K3r0 = K3_rec(1);
% subplot(1,3,3); 
% hold on; plot(0:size(K3_rec,2)-1,K3_rec(1,:)/K3r0,'LineWidth',linsz); hold off;
% % xlabel('\tau','fontsize',fntsz+2); ylabel('$\hat{k}_3$','Interpreter','Latex', 'fontsize',fntsz+5)
% set(gca, 'fontsize',fntsz);

hold off;
legend('True','Recovered','fontsize',14)
return;
%%

xstar = makePossibleComb({testinx',(1:n)',(1:n)'});

gpmodel.learn = false;
[hypers_learned, trnlml, Iout_z, Iout, Vout, filename] = run_gp_grid(gp_input, gpmodel,'params',params,'xstar',xstar,...
    'lognoise',-5,'numofstarts',50,'maxiteration',300,'filename',['bigdata',num2str(covs{2})],'hypers_init',bestinit);

%%
outImage = reshape(Iout,[n,n,5]);
figure(2)
for i = 1:length(testinx)
subplot(2,5,5+i); imagesc(outImage(:,:,i));
colormap(gray)
axis off
caxis([-30       63.932])
end
