function [] = run_fmriGP(filenames, savefilename, datafolder, savefolder, numberofsplits, maskfile)

if(nargin<2)
    savefilename = filenames{1};
end
start_time = datestr(now,30);
datafilename = [savefilename,start_time];

if(nargin<3)
    datafolder = '.';
end
if(nargin<4)
    savefolder = fullfile('.',datafilename);
    mkdir(savefolder);
end
if(nargin<5)
    numberofsplits=5;
end

% Check datasize of Runs
for filei = 1:length(filenames)
    try
       nii = load_nii([filenames{filei},'.nii']);
    catch err
        nii = load_untouch_nii([filenames{filei},'.nii']);
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

[ytr, yraw, mask, mask_wholebrain, hypers_detrend] = ...
    fmriGP_detrend(filenames, Fs, trialtime, savefilename, datafolder, savefolder, fullmask, 0.3);
disp('END DETREND');



%% plot trends
if(usejava('jvm'))
    %     save( fullfile(savefolder,datafilename) )
    
    
    masktrnd = zeros(size(mask));
%     masktrnd(3+(1:3),6+(1:3),12)=1;
masktrnd(20+(1:3),1+(1:3),16)=1;

    masktrnd = masktrnd.*mask;
    showrun =1;
    
    ytrnd = ytr(:,:,:,:,showrun);
    sz = size(ytrnd);
    ytrendmat = reshape(ytrnd,prod(sz(1:3)),prod(sz(4:end)));
    ytrendmat = ytrendmat(masktrnd(:)>0,:);
    ytrendmat = ytrendmat(1:min(3000,size(ytrendmat,1)),:);
    figure(7); plot(ytrendmat(2,:))
    
    
    ytrnd = yraw(:,:,:,:,showrun)-ytr(:,:,:,:,showrun);
    sz = size(ytrnd);
    ytrendmat = reshape(ytrnd,prod(sz(1:3)),prod(sz(4:end)));
    ytrendmat = ytrendmat(masktrnd(:)>0,:);
    ytrendmat = ytrendmat(1:min(3000,size(ytrendmat,1)),:);
    
    % % 3D display
    [X,Y] = meshgrid(1:size(ytrendmat,1),1:size(ytrendmat,2));
    figure(2); clf
%     plot3(X,Y,ytrendmat' - repmat(mean(ytrendmat'),256,1))
    plot(0:2:2*255,ytrendmat' - repmat(mean(ytrendmat'),256,1))
    % set(gca, 'YDir','reverse')
%     view(6,2)
    fntsz = 16;
%     zlabel('Intensity','fontsize',fntsz);
%     xlabel('Voxel index','fontsize',fntsz)
%     ylabel('time [sec]','fontsize',fntsz)
    ylabel('Drift Intensity','fontsize',fntsz);
    xlabel('Time [sec]','fontsize',fntsz)
%     view(90,0)
     axis('tight')
%     ylim([0,256])
    
    
    figure(4)
    Fs = 1/2;
    L = prod(sz(4:end));                     % Length of signal
    NFFT = 2^(0+nextpow2(L)); % Next power of 2 from length of y
    Yfft = fft(ytrendmat,NFFT,2)/L;
    f = Fs/2*linspace(0,1,NFFT/2+1);
    % Plot single-sided amplitude spectrum.
    semilogy(f,2*abs(Yfft(:,1:NFFT/2+1)))
    title('Single-Sided Amplitude Spectrum of y(t)')
    xlabel('Frequency (Hz)')
    ylabel('|Y(f)|')
%     xlim([0     0.010984])
    
    figure(5)
    ytrnd2 = yraw(:,:,:,:,showrun);
    sz = size(ytrnd2);
    ytrendmat2 = reshape(ytrnd2,prod(sz(1:3)),prod(sz(4:end)));
    ytrendmat2 = ytrendmat2(masktrnd(:)>0,:);
    ytrendmat2 = ytrendmat2(1:min(3000,size(ytrendmat2,1)),:);
    
    Fs = 1/2;
    L = prod(sz(4:end));                     % Length of signal
    NFFT = 2^(0+nextpow2(L)); % Next power of 2 from length of y
    Yrawfft = fft(ytrendmat2,NFFT,2)/L;
    f = Fs/2*linspace(0,1,NFFT/2+1);
    % Plot single-sided amplitude spectrum.
    semilogy(f,2*abs(Yrawfft(:,1:NFFT/2+1)))
    title('Single-Sided Amplitude Spectrum of y(t)')
    xlabel('Frequency (Hz)')
    ylabel('|Y(f)|')
    
     ytrnd2 = ytr(:,:,:,:,showrun);
    sz = size(ytrnd2);
    ytrendmat2 = reshape(ytrnd2,prod(sz(1:3)),prod(sz(4:end)));
    ytrendmat2 = ytrendmat2(masktrnd(:)>0,:);
    ytrendmat2 = ytrendmat2(1:min(3000,size(ytrendmat2,1)),:);
    
    Fs = 1/2;
    L = prod(sz(4:end));                     % Length of signal
    NFFT = 2^(0+nextpow2(L)); % Next power of 2 from length of y
    Ytrfft = fft(ytrendmat2,NFFT,2)/L;
 
    figure(6)
    vox=4
    semilogy(f,2*abs([Yrawfft(vox,1:NFFT/2+1); Yfft(vox,1:NFFT/2+1);  Ytrfft(vox,1:NFFT/2+1)]))
    title('Single-Sided Amplitude Spectrum of y(t)')
    xlabel('Frequency (Hz)')
    ylabel('|Y(f)|')
%     ylim([0,6])
%     xlim([0     0.010984])
    %%
    Y = zeros(prod(sz(1:3)),sz(4),4);
    for i=1:min(length(filenames),4)
        Y(:,:,i) = reshape(yraw(:,:,:,:,i)-ytr(:,:,:,:,i),prod(sz(1:3)),sz(4));
    end
    figure(3)
    showRMS(Y, fullfile('C:\Users\Elad\Documents\Research\FMRI\88','88_orig_on_TRIO_Y_NDC_333.4dfp.nii'), 1, sz, [10 10],mask_wholebrain,0.5*(1+masktrnd));
    
    clear ytrendmat Y;
end

%%
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
%%
numofparalleljobs = 38;
make_fmri_parallel_jobs(datafilename, numofparalleljobs, savefolder, savefolder)
cd(savefolder);
system('chmod 777 batch_run')
system('./batch_run')
