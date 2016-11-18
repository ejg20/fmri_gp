clear variables;
close all;

NUMBEROFPARALLELJOBS = 20;

test = 'TASK';

addpath(genpath('/home/research/gilboae/fMRI'));
addpath(genpath('/home/research/gilboae/gp'));

savefilename = '88_100316CR_1-pol_20140129T132253';
load(fullfile(savefilename),'ytr','mask','parcellateMat','trialsamp',...
    'blkstartx','blkstarty','blkstartz','blkendx','blkendy','blkendz');

%datafolder = fullfile('C:','Users','Elad','Documents','Research','FMRI','88','bigregion');
%savefolder = fullfile('C:','Users','Elad','Documents','Research','FMRI','88','bigregion');
 datafolder = fullfile('.');
 savefolder = fullfile('.');


%% Get number from filename
runname = mfilename;
runnum = str2double(runname(isstrprop(runname,'digit')));
disp(runnum)

%% Split for parallel runs
blkindx = 1:size(parcellateMat,1);
parjobindx = round(linspace(0,size(parcellateMat,1),NUMBEROFPARALLELJOBS+1));
blkindx = blkindx(parjobindx(runnum)+1:parjobindx(runnum+1));

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
        ytrmat = reshape(ytr_blk,prod(sz(1:3)),trialsamp,prod(sz(4:end))/trialsamp);
        ytrmatext = repmat(mean(ytrmat,3),1,prod(sz(4:end))/trialsamp);
        ytrmatint = reshape(ytrmat,size(ytrmatext)) - ytrmatext;
        s2ytrint = var(ytrmatint(mask_blk(:),:),[],2);
        s2ytrint = min(s2ytrint,100);   %cap allowed noise are 100 for convergence
        sampleinitnoise = false;
        lambda = 0;

        clear ytrmatint ytrmatext ytrmat

        [hypers_learned, gp_input, gpmodel, lognoise, noise_std_mat] = ...
            fmriGP_learn_sampnoise(ytr_blk, mask_blk, lambda, s2ytrint,sampleinitnoise);

    %     [hypers_learned, gp_input, gpmodel, lognoise, noise_std_mat] = ...
    %         fmriGP_learn(ytr_blk, mask_blk, find(~isnan(ytr_blk)), lambda, s2ytrint);

        %     save( fullfile(savefolder,[savefilename,start_time]) )

    %     figure(3);
    %     fmriGP_showkernel(ytr_blk, hypers_learned, gp_input, gpmodel);

        [yout] = fmriGP_denoise(gp_input, gpmodel, ytr_blk, hypers_learned);

        % calc extrinsic
        if(strcmp(test,'TASK'))
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
        else

            yext = zeros(size(yout));
            yint = yout;

        end

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
%%
filename = 'fmriGP_script_wholebrain_p';

fid = fopen('batch_run', 'w');
fprintf(fid, '#!/bin/sh\n');
fprintf(fid, '#$ -cwd\n');

for i =1:NUMBEROFPARALLELJOBS
    system(['copy ',filename,'.m ',filename,num2str(i),'.m']);
    fprintf(fid, ['nohup /cluster/cloud/matlab/bin/matlab -nodisplay -nojvm <', filename,num2str(i),...
        '.m > test',num2str(i),'.out&\n']);
end

fclose(fid)



