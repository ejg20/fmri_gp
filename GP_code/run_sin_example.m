close all;
clear variables;

global prodcovfunc ;
prodcovfunc = 'gp_grid_prodcov_v5';  
alpha_prev = 0;

D = 2;
gamma2 = 0.001;
mu_f = 0;

% t = (0:1:100-1)';
Fs = 200;
T = 1/Fs;                     % Sample time
L = 50;                     % Length of signal
t = (0:1:L)'*T;                % Time vector

[X,Y] = meshgrid(t, t);

draw1 = (1*sin(2*pi.*X*3)+sin(2*pi.*X*18)).*sin(2*pi.*Y*6);
draw2 = (1*sin(2*pi.*X*6)+sin(2*pi.*X*10)).*sin(2*pi.*Y*15);
ytr = draw1+draw2+gamma2*randn(size(draw1));



mask = ones(size(X));
% mask(50:90,50:90) = 0;
mask(10:40,10:40) = 0;

gp_input = gp_grid_input_class(ytr,find(mask >0),Fs);

noise_struct = gp_grid_noise_class(ones(size(ytr)),gp_input.index_to_N);
noise_struct.learn = true;

% covs = {'covSM1D',20};
covs{1} = {{'covSM1D',20},{'covSM1D',20}};
% covs{2} = {{'covSM1D',20},{'covSM1D',20}};
% covs{3} = {{'covSM1D',1},{'covSM1D',1}};
% covs{4} = {{'covSM1D',1},{'covSM1D',1}};
% covs{5} = {{'covSM1D',1},{'covSM1D',1}};
% covs{6} = {{'covSM1D',1},{'covSM1D',1}};
% covs{7} = {{'covSM1D',1},{'covSM1D',1}};
% covs{8} = {{'covSM1D',1},{'covSM1D',1}};


gpmodel = gp_grid_gpmodel_class();
gpmodel.cov = covs;
gpmodel.noise_struct = noise_struct;
gpmodel.hyps_in_d  = make_hyps_in_d_v2(covs);
% gpmodel.hyps_in_d  = make_hyps_in_d([20,20],covs);
gpmodel.learn = true;

Z = length(gpmodel.cov);
P = length(gpmodel.cov{1});

lambda = 1e6;
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
        


xstar = [];
params.wm = std(gp_input.zeromeandata);
Gs = gp_grid_size(gp_input.xgrid);
params.sm = 2*Gs./gp_input.Fs;

data_for_plot = zeros(size(X));
data_for_plot(gp_input.index_to_N) = gp_input.get_data();

figure(1)
subplot(1,2,1); CAX = showPatternPlot(data_for_plot,[],'Original Pattern');
freezeColors;
cbfreeze(colorbar);
subplot(1,2,2); showPatternSpectrumPlot(data_for_plot,[],'Log magnitude spectrum',Fs,[-20,20,-20,20]);
drawnow;

% [hypers_learned, trnlml, Iout_z, Iout, Vout, filename] = run_gp_grid(gp_input, gpmodel,'params',params,'xstar',xstar,...
%     'lognoise',-5,'numofstarts',20,'maxiteration',500);
[hypers_learned, trnlml, Iout_z, Iout, Vout, filename] = run_gp_grid_v2(gp_input, gpmodel,'params',params,'xstar',xstar,...
    'lognoise',-5,'numofstarts',50,'maxiteration',200,'filename',['sin',num2str(covs{1}{1}{2})]);
bestinit = hypers_learned;


% showPlotsKernel(data_for_plot_ext,true_hyps,make_gpsmp_hyps_in_d(Qs),cov,CAX,Fs,[-20,20,-20,20],1, 'Trained Pattern');
% close 2
figure(2);
for z = 1:Z
    showSMkernel_v2(hypers_learned, {gpmodel.hyps_in_d{z}}, gp_input.Fs)
    colorbar
    drawnow;
    pause;
end

% figure(2); showSMkernel(hypers_learned, gpmodel.hyps_in_d, gp_input.Fs);
% axis([0,20,0,20])

% figure(3); showSMkernel(hypers_learned, gpmodel.hyps_in_d{2}, gp_input.Fs)
% axis([0,20,0,20])

% return

%%
% [Xs Ys] = meshgrid((0:300)'*T,(0:300)'*T);
% xstar= [Xs(:),Ys(:)];
% xstar(max(xstar,[],2) <=150*T,:)=[];

xstar = gp_input.make_xstar(find(mask == 0));

gpmodel.learn = false;
[hypers_learned2, trnlml, Iout_z, Iout, Vout, filename] = run_gp_grid_v2(gp_input, gpmodel,'params',params,'xstar',xstar,...
    'lognoise',-5,'numofstarts',10,'maxiteration',500,'filename',['sin',num2str(covs{1}{1}{2})],'hypers_init',hypers_learned);

% dataforplot = NaN(size(Xs));
% dataforplot(1:151,1:151) = ytr;
% [Xs Ys] = meshgrid((0:300),(0:300));
% xstar= [Xs(:),Ys(:)];
% 
% dataforplot(max(xstar,[],2) >150)=Iout;
% figure; imagesc(dataforplot)

dataforplot =  ytr;
dataforplot(mask == 0) = Iout;
figure; imagesc(dataforplot)
