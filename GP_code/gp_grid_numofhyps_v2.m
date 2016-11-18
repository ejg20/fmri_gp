function nhyps = gp_grid_numofhyps_v2(hyps_in_d)
    
Z = length(hyps_in_d);

nhyps = 0;
for z = 1:Z
    z_nhyps = length(unique(cell2mat(hyps_in_d{z})));
    nhyps = nhyps+z_nhyps;
end

