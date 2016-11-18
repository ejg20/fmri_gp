likfunc = @likGauss;
covfunc = @gpr_covSEard_grid; hyp.cov = log(2.2*rand(D,1)); hyp.lik = log(0.1);
%disp('hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfunc, likfunc, x, y)');
hyp = minimize(hyp, @gp, -100, @infExact, [], covfunc, likfunc, x, y);
disp(' ');