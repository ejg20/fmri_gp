function [] = run_fmri_parallel_jobs(jobindex, savefilename, number_of_jobs, datafolder, savefolder)

if(nargin < 3 || isempty(number_of_jobs))
    number_of_jobs = 20;
end

if(isempty(jobindex))
    % Get number from filename
    runname = mfilename;
    jobindex = str2double(runname(isstrprop(runname,'digit')));
end

disp(jobindex)

if(nargin< 4)
    datafolder = '.';
end
if(nargin<5)
    savefolder = '.';
end

% savefilename = '88_100316CR_1-pol_20140129T132253';
load(fullfile(datafolder,savefilename),'ytr','mask','parcellateMat','trialsamp',...
    'blkstartx','blkstarty','blkstartz','blkendx','blkendy','blkendz');


%% Split for parallel runs
blkindx = 1:size(parcellateMat,1);
parjobindx = round(linspace(0,size(parcellateMat,1),number_of_jobs+1));
blkindx = blkindx(parjobindx(jobindex)+1:parjobindx(jobindex+1));

%%
blkfolder = fullfile(savefolder,[savefilename,'_blocks']);
mkdir(blkfolder);


for blknum = blkindx
    try
        blkx = parcellateMat(blknum,1);
        blky = parcellateMat(blknum,2);
        blkz = parcellateMat(blknum,3);
        mask_blk = mask(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
            blkstartz(blkz):blkendz(blkz));
        ytr_blk = ytr(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
            blkstartz(blkz):blkendz(blkz), :, :);
        
        sz = size(ytr_blk);
        % emperical estimation of the noise
        % NEED TO ADD THE IMPORTANT CASE WHERE THE TRIALS ARE IN THE DESIGN
        % MATRIX!!!!!
        if(prod(sz(4))/trialsamp == round(prod(sz(4))/trialsamp))
            % this is when using a periodic constant trial
            ytrmat = reshape(ytr_blk,prod(sz(1:3)),trialsamp,prod(sz(4:end))/trialsamp);
            ytrmatext = repmat(mean(ytrmat,3),1,prod(sz(4:end))/trialsamp);
            ytrmatint = reshape(ytrmat,size(ytrmatext)) - ytrmatext;
            s2ytrint = var(ytrmatint(mask_blk(:),:),[],2);
            s2ytrint = min(s2ytrint,100);   %cap allowed noise are 100 for convergence
        else
            % this is when you do not know anything about the trials then
            % you can only estimate the noise base on the std
            ytrmatint = reshape(ytr_blk,prod(sz(1:3)),prod(sz(4:end)));
            s2ytrint = var(ytrmatint(mask_blk(:),:),[],2);
            s2ytrint = min(s2ytrint,30);   %cap allowed noise are 100 for convergence
        end
        sampleinitnoise = false;
        lambda = 0;
        
        clear ytrmatint ytrmatext ytrmat
        
        for trytolearn = 1:5
        [hypers_learned, gp_input, gpmodel, lognoise, noise_std_mat] = ...
            fmriGP_learn_sampnoise(ytr_blk, mask_blk, lambda, s2ytrint, sampleinitnoise);

        %     [hypers_learned, gp_input, gpmodel, lognoise, noise_std_mat] = ...
        %         fmriGP_learn(ytr_blk, mask_blk, find(~isnan(ytr_blk)), lambda, s2ytrint);
        
        %     save( fullfile(savefolder,[savefilename,start_time]) )
        
        %     figure(3);
        %     fmriGP_showkernel(ytr_blk, hypers_learned, gp_input, gpmodel);
        
        [yout] = fmriGP_denoise(gp_input, gpmodel, ytr_blk, hypers_learned);
       
        %check that the inference is valid. mean absolute intensity Shouldn't
        %be much bigger than original; 3 is arbitrary
        if(mean(abs(yout(:)))< 3*mean(abs(ytr_blk(:))))
            break
        end
        end
        if(mean(abs(yout(:)))> 3*mean(abs(ytr_blk(:))))
            error('could not learn');
        end
       
        % calc extrinsic
        
        sz = size(ytr_blk);
        ytrmat = reshape(ytr_blk,prod(sz(1:3)),trialsamp,prod(sz(4:end))/trialsamp);
        ytrmatext = repmat(mean(ytrmat,3),1,prod(sz(4:end))/trialsamp);
        sz = size(yout);
        youtmat = reshape(yout,prod(sz(1:3)),trialsamp,prod(sz(4:end))/trialsamp);
        youtmatext = repmat(mean(youtmat,3),1,prod(sz(4:end))/trialsamp);
        
        %         figure(2);
        %         vox=1;
        %         indx  = find(mask_blk>0);
        %         plot(ytrmatext(indx(vox),:),':'); hold on;
        %         plot(youtmatext(indx(vox),:),'r'); hold off;
        
        yext = reshape(youtmatext,size(ytr_blk));
        
        yint = yout - yext;
        
        clear youtmatext youtmat ytrmat
        
        
        yres = ytr_blk - yout;
        
        blk = struct('hypers_learned',hypers_learned,...
            'lognoise',lognoise, 'noise_std_mat',noise_std_mat,...
            'yext',yext,'yint',yint);
        
        save( fullfile(blkfolder,['blk',num2str(blknum)]), 'blk')
        
    catch err
        save( fullfile(blkfolder,['blk',num2str(blknum),'_err']), 'err')
    end
    
    
end

return



