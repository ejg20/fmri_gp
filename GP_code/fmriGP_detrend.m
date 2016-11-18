function [ytr_detrend, y_raw, mask, mask_wholebrain, hypers_learned]...
    = fmriGP_detrend(filenames, Fs, trialtime, savefilename, datafolder, savefolder, fullmask, ratioforsample)

if(nargin<7)
    fullmask = [];
end
if(nargin<8)
    ratioforsample = 0;
end


multi_index_to_N_cell = [];
mask=[];
ytr_detrend=[];
y_raw=[];
multi_index_to_N=[];

for fi = 1:length(filenames)
    filename = filenames{fi};
    nii_filename = fullfile(datafolder,filename);
    % nii_filename = 'sphere/4_88_100521CR_6-gainfield-down_on_TRIO_Y_NDC_333_MASK_lh-V1dorsal_sphere12mm';
    
    %     [header, ext, filetype, machine] = load_untouch_header_only([nii_filename,'.nii']);
    try
        nii = load_nii([nii_filename,'.nii']);
    catch err
        nii = load_untouch_nii([nii_filename,'.nii']);
    end
    nii.img = single(nii.img);
    if(~isempty(fullmask))
        nii.img = nii.img.*repmat(fullmask,[1,1,1,size(nii.img,4)]);
    end
    
    sumtime = sum(abs(nii.img),4);
    ix = sum(squeeze(sum(sumtime,3)),2)>0;
    iy = sum(squeeze(sum(sumtime,3)),1)>0;
    iz = sum(squeeze(sum(sumtime,1)),1)>0;
    mask_wholebrain = {ix,iy,iz};
    
    
    %     tindx = 1:size(nii.img,4);
    %     tmp = reshape(tindx, 32, length(tindx)/32);
    %     tmp(:,3:3:end) = [];
    %     tindx = tmp(:);
    %     ytr= nii.img(ix,iy,iz,tindx);
    ytr= nii.img(ix,iy,iz,:);
    
    if(isempty(y_raw))
        y_raw =  zeros([size(ytr),length(filenames)]);
    end
    y_raw(:,:,:,:,fi) = ytr;
    
    mask = squeeze(sum(sum(abs(y_raw),5),4))>0;
    
    %     if(false)
    %         for t = 1:size(ytr,4)
    %             timesample = ytr(:,:,:,t);
    %             [m n o]=size(timesample);
    %             figure(1)
    %             [x,y,z] = meshgrid(1:m,1:n,1:o);
    %             nonzeroindex = find(timesample(:)>0);
    %             scatter3(x(nonzeroindex),y(nonzeroindex),z(nonzeroindex),90,timesample(nonzeroindex),'.')
    %             drawnow
    %         end
    %     end
    
    %%
    
    if(isempty(Fs))
        Fs = [3e-3,3e-3,3e-3, 2].^(-1);
    end
    
    gp_input = gp_grid_input_class(ytr,find(abs(ytr)>0),Fs);
    
    
    %%%%%%%%%%%%%%%%%%%%% LEARN %%%%%%%%%%%%%%%%%%%%%%%%
    
    if(ratioforsample==0 || ratioforsample==1)
        gp_input_samp = gp_input;
        ytr_samp = ytr;
    else
        
        % make a random subgrid to sample for the trend kernel
        
        % Try 20 time to get the samples from a complete grid
        for numoftries = 1:1000
            %
            sampgridx = randperm(sum(ix));
            sampgridy = randperm(sum(iy));
            sampgridz = randperm(sum(iz));
            sampgridx = sort(sampgridx(1:round(sum(ix)*ratioforsample)));
            sampgridy = sort(sampgridy(1:round(sum(iy)*ratioforsample)));
            sampgridz = sort(sampgridz(1:round(sum(iz)*ratioforsample)));
            
            sampgridprojtime = sum(abs(ytr(sampgridx,sampgridy,sampgridz,:)),4);
            if(isempty(find(sampgridprojtime(:) == 0,1)))
                break;
            end
        end
        if(~isempty(find(sampgridprojtime(:) == 0,1)))
            disp('Could not sample a complete grid');
        else
            disp('Found a complete grid sample');
        end
        ytr_samp = ytr(sampgridx,sampgridy,sampgridz,:);
        gp_input_samp = gp_grid_input_class(ytr_samp,1:numel(ytr_samp),Fs);
        subxgrid = cell(1,gp_input.get_P);
        subxgrid{4} = gp_input.xgrid{1};
        subxgrid{1} = sampgridx/Fs(1);
        subxgrid{2} = sampgridy/Fs(2);
        subxgrid{3} = sampgridz/Fs(3);
        gp_input_samp.make_xgrid(subxgrid{:})
    end
    
    noise_struct = gp_grid_noise_class(ones(size(ytr_samp)),gp_input_samp.index_to_N);
    noise_struct.learn = true;
    
    covs{1} = {{'covSEard'},{'covSEard'},{'covSEard'},{'covSEard'}};
    %     covs{1} = {{'covSM1D',1},{'covSM1D',1},{'covSM1D',1},{'covSM1D',1}};
    
    gpmodel = gp_grid_gpmodel_class();
    gpmodel.cov = covs;
    gpmodel.noise_struct = noise_struct;
    gpmodel.hyps_in_d  = make_hyps_in_d_v2(covs);
    % gpmodel.hyps_in_d  = make_hyps_in_d([20,20],covs);
    gpmodel.learn = true;
    
    
    xstar = [];
    params.wm = std(gp_input_samp.zeromeandata);
    %     Gs = gp_grid_size(gp_input.xgrid);
    params.sm = [200,1e-10,1e-10,1e-10];
    
    params.prodcovfunc = 'gp_grid_prodcov_v2';
    params.predfunc = 'gpgrid_predict_fullkron_denoise';
    
    %% GP
    highperiodthresh = round(trialtime*1.5); % Gaussian low pass, 2 std = 1.5 trial time period.
    for numoftries = 1:2
        [hypers_learned, ~, ~, ~, ~, ~] = run_gp_grid_v2(gp_input_samp, gpmodel,'params',params,'xstar',xstar,...
            'lognoise',2,'numofstarts',20,'maxiteration',300);
        bestinit = hypers_learned;
        if(exp(hypers_learned(1)) > highperiodthresh)
            break;
        end
    end
    
    %     bestinit([3,5,7]) = min(bestinit([3,5,7]),-10);
    bestinit(1) = max(bestinit(1),log(highperiodthresh));
    
    clear gp_input_samp ytr_samp
    
    %%%%%%%%%%%%%%%%%%%%% SMOOTH %%%%%%%%%%%%%%%%%%%%%%%%
    
    % Treat brain as homogenous for better computational complexity
    
    %     noise_struct = gp_grid_noise_class(ones(size(ytr)),gp_input.index_to_N);
    N = numel(ytr);
    gp_input_full = gp_grid_input_class(ytr,(1:N)',Fs);
    noise_struct = gp_grid_noise_class(ones(size(ytr)),(1:N)');
    noise_struct.learn = true;
    gpmodel.noise_struct = noise_struct;
    
    gpmodel.learn = false;
    xstar.index_to_N = (1:N)';
    xstar.subs= gp_input_full.make_xstar(xstar.index_to_N);
    [~, ~, ~, Iout_full, ~, ~] = run_gp_grid_v2(gp_input_full, gpmodel,'params',params,'xstar',xstar,...
        'lognoise',0,'numofstarts',10,'maxiteration',500, 'hypers_init', bestinit);
    
    
    %% SHOW VOXELS
    yout = ytr;
    yout(gp_input.index_to_N) = Iout_full(gp_input.index_to_N);
    [xt,yt,zt] = find(mask>0);
    
    if(usejava('jvm'))
        %     h=figure(1);
        h=figure;
        Q = 4;
        for vox = 1:Q
            subplot(Q,2,2*vox-1); plot([squeeze(ytr(xt(vox),yt(vox),zt(vox),:,1)),squeeze(yout(xt(vox),yt(vox),zt(vox),:,1))]);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%% DELETE SMOOTH %%%%%%%%%%%%%%%%%%%%%%%%
    
    if(isempty(ytr_detrend))
        ytr_detrend =  zeros([size(ytr),length(filenames)]);
    end
    %     tmp = nan(size(ytr)); tmp(gp_input.index_to_N) = ytr(gp_input.index_to_N) - Iout_full(gp_input.index_to_N);
    %     ytr_detrend(:,:,:,:,fi) = tmp;
    tmp = nan(size(ytr)); tmp(:) = ytr(:) - Iout_full(:);
    ytr_detrend(:,:,:,:,fi) = tmp;
    
    if(usejava('jvm'))
        %     figure(1);
        Q = 4;
        for vox = 1:Q
            subplot(Q,2,2*vox); plot(squeeze(ytr_detrend(xt(vox),yt(vox),zt(vox),:,fi)));
        end
        print(h,'-djpeg',fullfile(savefolder,[savefilename,'_notrend']));
    end
    %     multi_index_to_N = [multi_index_to_N;(fi-1)*numel(ytr)+gp_input.index_to_N];
    %     multi_index_to_N_cell{fi} = gp_input.index_to_N;
end

%% SAVE

% save(fullfile(savefolder,['FMRI_GP1_',savefilename]));
% print(h,'-djpeg',fullfile(savefolder,[savefilename,'_trend']))