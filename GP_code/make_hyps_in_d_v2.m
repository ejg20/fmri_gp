function hyps_in_d = make_hyps_in_d_v2(covs)
Z = length(covs);
P = length(covs{1});
D = 1;
hyps_in_d = cell(Z,1);
hyps_in_dz = cell(P,1);

runingindex = 0;
for z = 1:Z
    for p = 1:P
        numofhypers = eval(feval(covs{z}{p}{:}));
        indxs = (1:numofhypers)';
        hyps_in_dz{p} = runingindex+indxs;
        runingindex = runingindex+numofhypers;
    end
    hyps_in_d{z} = hyps_in_dz;
end