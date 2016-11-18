%function [] = run_fmriGP(filenames, savefilename, datafolder, savefolder, numberofsplits, maskfile)
%
% must install libraries:
%   gpml-matlab (used v3.2-2013-01-15) http://www.gaussianprocess.org/gpml/code/
%   NIFTI matlab (used NIFTI_20130306) included
%

close all
%test
nargin= 0;

if(nargin<1)
    datafolder = './data/';
    filenames = {   '' };
    savefolder = './data/results/';
    mkdir(datafolder);
    mkdir(savefolder);
end

if(nargin<2)
    savefilename = filenames{1};
end
start_time = datestr(now,30);
datafilename = [savefilename,start_time];

if(nargin<3 && ~exist('datafolder','var'))
    datafolder = '.';
end
if(nargin<4  && ~exist('savefolder','var'))
    savefolder = fullfile('.',datafilename);
    mkdir(savefolder);
end
if(nargin<5)
    numberofsplits=5;
end

% Check datasize of Runs
for filei = 1:length(filenames)
    try
        nii = load_nii([datafolder,filenames{filei},'.nii']);
    catch err
        nii = load_untouch_nii([datafolder,filenames{filei},'.nii']);
    end
    
    if(filei == 1)
        sz = size(nii.img);
    else
        if prod(double(size(nii.img) ~= sz))
            error('Nifty files are not the same size')
        end
    end
end

% Check datasize of Mask
if nargin < 6
    fullmask = ones([sz(1:3),1]); %no mask
else
    try
        masknii = load_nii([maskfile,'.nii'] );
    catch err
        masknii = load_untouch_nii([maskfile,'.nii'] );
    end
    
    fullmask = masknii.img;
    if prod(double((size(fullmask) ~= sz(1:3))))
        error('Mask file are not the same size as runs')
    end
end

Fs = [nii.hdr.dime.pixdim(2:4)*1e-3, 2].^(-1);
trialsamp = 32;
trialtime = trialsamp/Fs(4);

clear masknii;
%% detrend data


%% if want to try to detrend using GP
[ytr, yraw, mask, mask_wholebrain, hypers_detrend] = ...
    fmriGP_detrend(filenames, Fs, trialtime, savefilename, datafolder, savefolder, fullmask, 0.3);
disp('END DETREND');


%% Make blocks
sz = size(ytr);

blksize = ceil(sz(1:3)/numberofsplits);
blksidx = round(linspace(0,sz(1),numberofsplits+1));
blksidy = round(linspace(0,sz(2),numberofsplits+1));
blksidz = round(linspace(0,sz(3),numberofsplits+1));
blkstartx = blksidx(1:end-1)+1; blkendx = blksidx(2:end);
blkstarty =  blksidy(1:end-1)+1; blkendy = blksidy(2:end);
blkstartz =  blksidz(1:end-1)+1; blkendz = blksidz(2:end);
parcellateMat = makePossibleComb({1:length(blkstartx),...
    1:length(blkstarty),1:length(blkstartz)});

blksidxmid = round(blksize(1)/2)+blksidx(1:end-1);
blksidymid = round(blksize(2)/2)+blksidy(1:end-1);
blksidzmid = round(blksize(3)/2)+blksidz(1:end-1);
blkstartxmid = blksidxmid(1:end-1)+1; blkendxmid = blksidxmid(2:end);
blkstartymid =  blksidymid(1:end-1)+1; blkendymid = blksidymid(2:end);
blkstartzmid =  blksidzmid(1:end-1)+1; blkendzmid = blksidzmid(2:end);

parcellateMatmid = makePossibleComb({
    length(blkstartx)+(1:length(blkstartxmid)),...
    length(blkstarty)+(1:length(blkstartymid)),...
    length(blkstartz)+(1:length(blkstartzmid))});

blkstartx = [blkstartx blkstartxmid];
blkstarty = [blkstarty blkstartymid];
blkstartz = [blkstartz blkstartzmid];

blkendx = [blkendx blkendxmid];
blkendy = [blkendy blkendymid];
blkendz = [blkendz blkendzmid];

parcellateMat = [parcellateMat;parcellateMatmid];

fullfile(savefolder,datafilename)
save(  fullfile(savefolder,datafilename) );

disp('END SAVE');


%% Can Run Jobs in parallel if server exists using -- run_fmri_parallel_jobs(jobindex, savefilename, number_of_jobs, datafolder, savefolder)

%% if runing in series (can take a long time) run code bellow

%% Split for parallel runs
blkindx = 1:size(parcellateMat,1);
%parjobindx = round(linspace(0,size(parcellateMat,1),number_of_jobs+1));
%blkindx = blkindx(parjobindx(jobindex)+1:parjobindx(jobindex+1));

%%
blkfolder = fullfile(savefolder,[savefilename,'_blocks']);
mkdir(blkfolder);
blkindx

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
        %        fmriGP_learn(ytr_blk, mask_blk, find(~isnan(ytr_blk)), lambda, s2ytrint);
        
        %     save( fullfile(savefolder,[savefilename,start_time]) )
        
        %     figure(3);
        %     fmriGP_showkernel(ytr_blk, hypers_learned, gp_input, gpmodel);
        
        [yout] = fmriGP_denoise(gp_input, gpmodel, ytr_blk, hypers_learned);
        
        % calc extrinsic

        % if(strcmp(test,'TASK'))
        %     sz = size(ytr_blk);
        %     ytrmat = reshape(ytr_blk,prod(sz(1:3)),trialsamp,prod(sz(4:end))/trialsamp);
        %     ytrmatext = repmat(mean(ytrmat,3),1,prod(sz(4:end))/trialsamp);
        %     sz = size(yout);
        %     youtmat = reshape(yout,prod(sz(1:3)),trialsamp,prod(sz(4:end))/trialsamp);
        %     youtmatext = repmat(mean(youtmat,3),1,prod(sz(4:end))/trialsamp);
        %     
        %     %         figure(2);
        %     %         vox=1;
        %     %         indx  = find(mask_blk>0);
        %     %         plot(ytrmatext(indx(vox),:),':'); hold on;
        %     %         plot(youtmatext(indx(vox),:),'r'); hold off;
        %     
        %     yext = reshape(youtmatext,size(ytr_blk));
        %     
        %     yint = yout - yext;
        %     
        %     clear youtmatext youtmat ytrmat
        % else
            
            yext = zeros(size(yout));
            yint = yout;
            
        % end
        
        yres = ytr_blk - yout;
        
        blk = struct('hypers_learned',hypers_learned,...
            'lognoise',lognoise, 'noise_std_mat',noise_std_mat,...
            'yext',yext,'yint',yint);
        
        parsave( fullfile(blkfolder,['blk',num2str(blknum)]), blk)
        
    catch err
        parsave( fullfile(blkfolder,['blk',num2str(blknum),'_err']), err)
    end
    
end

return