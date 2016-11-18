function [tr, tst, cvxstar] = old_gp_grid_set_splitsets(tstset, input, gpmodel)
D = length(input.xgrid);
Gs = cell(D,1);
for d = 1:D
    Gs{d} = 1:length(input.xgrid{d});
end

[cvxstar_indx, ~, cvindx_to_N] = intersect(tstset,input.index_to_N);

cvxstar = input.make_xstar(cvxstar_indx);

% possible_cvxstar_sub = makePossibleComb(Gs);
% cvxstar_sub = possible_cvxstar_sub(cvxstar_indx,:);

% instantiate the two sets to the original input and gpmodel
% tr.input = gp_grid_input_class(input);
% tst.input = gp_grid_input_class(input);
tr.gpmodel = gpmodel;
tst.gpmodel = gpmodel;

% tr.input.xgrid = input.xgrid;
% tst.input.xgrid = input.xgrid;

% tst.input.index_to_N = cvxstar_indx;  % for simplicity, relate to full matrix
% tst.input.data = input.data(cvindx_to_N);
tst.input = gp_grid_input_class(input.get_data(),cvxstar_indx,input.Fs);
tst.input.copy_xgrid(input);

tr_index_to_N = input.index_to_N;
tr_index_to_N(cvindx_to_N) = [];   %takeout indices that are in cv
% tr.input.data = input.data;
% tr.input.data(cvindx_to_N) = [];   %takeout indices that are in cv
tr.input = gp_grid_input_class(input.get_data(),tr_index_to_N,input.Fs);
tr.input.copy_xgrid(input);

tst.gpmodel.noise_struct.learn = gpmodel.noise_struct.learn;
tst.gpmodel.noise_struct.sphericalNoise = false;
tst.gpmodel.noise_struct.var = gpmodel.noise_struct.var;

tr.gpmodel.noise_struct.learn = gpmodel.noise_struct.learn;
tr.gpmodel.noise_struct.sphericalNoise = false;
tr.gpmodel.noise_struct.var = gpmodel.noise_struct.var;

end