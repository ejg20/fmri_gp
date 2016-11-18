function [yout] = fmriGP_denoise(gp_input, gpmodel, ytr, bestinit)

params.prodcovfunc = 'gp_grid_prodcov_v2_hetero';
params.predfunc = 'gpgrid_predict_fullkron_denoise';

gpmodel.learn = false;
xstar.index_to_N = gp_input.index_to_N;
xstar.subs= gp_input.make_xstar(xstar.index_to_N);
[~, ~, ~, Iout, ~, ~] = run_gp_grid_v2(gp_input, gpmodel,'params',params,'xstar',xstar,...
    'lognoise',0,'numofstarts',10,'maxiteration',500, 'hypers_init', bestinit);

yout = ytr;
yout(gp_input.index_to_N) = Iout;
