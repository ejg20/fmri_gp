BIAS = 100; %for neurosicence programs. if too close to 0 it doesnt work
%addpath(genpath('/home/research/gilboae/fMRI'))
%addpath(genpath('/home/research/gilboae/gp'))
datafolder = 'data/results/';
filename = filenames{1};
blkdatafolder = [datafolder,filename,'_blocks'];


%load([datafolder,filename],'ytr','mask', 'filenames','datafolder','savefolder','parcellateMat',....
%    'blkstartx','blkstarty','blkstartz',...
%    'blkendx','blkendy','blkendz')
sz = size(ytr);

% niidatafolder = datafolder;
% niisavefolder = savefolder;
nii_filenames = filenames;

niisavefolder = './data/somatotopy_data_foot';
niidatafolder = './data/somatotopy_data_foot';


if(0)
    Z = 5;
    blksize = ceil(sz(1:3)/Z);
    blksidx = round(linspace(1,sz(1),Z+1));
    blksidy = round(linspace(1,sz(2),Z+1));
    blksidz = round(linspace(1,sz(3),Z+1));
    blkstartx = blksidx(1:end-1); blkendx = blksidx(2:end);
    blkstarty =  blksidy(1:end-1); blkendy = blksidy(2:end);
    blkstartz =  blksidz(1:end-1); blkendz = blksidz(2:end);
    parcellateMat = makePossibleComb({1:length(blkstartx),...
        1:length(blkstarty),1:length(blkstartz)});
    
    blksidxmid = round(blksize(1)/2)+blksidx(1:end-1);
    blksidymid = round(blksize(2)/2)+blksidy(1:end-1);
    blksidzmid = round(blksize(3)/2)+blksidz(1:end-1);
    blkstartxmid = blksidxmid(1:end-1); blkendxmid = blksidxmid(2:end);
    blkstartymid =  blksidymid(1:end-1); blkendymid = blksidymid(2:end);
    blkstartzmid =  blksidzmid(1:end-1); blkendzmid = blksidzmid(2:end);
    
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
    
    ytr_gp_ext = zeros(size(ytr));
    ytr_gp_int = zeros(size(ytr));
    ytr_gp= zeros(size(ytr));
end

ytr_gp= ytr;
counter = zeros(size(mask));
noise_gp = zeros(size(ytr(:,:,:,1,1)));


for blknum = 1:length(parcellateMat)
   disp(blknum)
    try
        load(fullfile(blkdatafolder,['blk',num2str(blknum)]))
    catch err
        warning(err.message)
        continue
    end
    blkx = parcellateMat(blknum,1);
    blky = parcellateMat(blknum,2);
    blkz = parcellateMat(blknum,3);
    mask_blk = mask(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
        blkstartz(blkz):blkendz(blkz));
%     ytr_gp_ext(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
%         blkstartz(blkz):blkendz(blkz), :, :) = ...
%         ytr_gp_ext(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
%         blkstartz(blkz):blkendz(blkz), :, :) + blk.yext;
%     ytr_gp_int(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
%         blkstartz(blkz):blkendz(blkz), :, :) = ...
%         ytr_gp_int(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
%         blkstartz(blkz):blkendz(blkz), :, :) + blk.yint;
    
    ytr_gp(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
        blkstartz(blkz):blkendz(blkz), :, :) = ...
        ytr_gp(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
        blkstartz(blkz):blkendz(blkz), :, :) + blk.yint+blk.yext;

    % the counter is to average overlapping areas
    counter(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
        blkstartz(blkz):blkendz(blkz)) = ....
        counter(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
        blkstartz(blkz):blkendz(blkz))+mask_blk;
    
    ytr_blk = ytr(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
        blkstartz(blkz):blkendz(blkz), :, :);
    
    
    blk_noise_gp = zeros(size(blk.yext(:,:,:,1,1)));
    blk_noise_gp(~isnan(blk.yext(:,:,:,1,1))) = blk.hypers_learned((end-length(blk.noise_std_mat)+1):end);
    noise_gp(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
        blkstartz(blkz):blkendz(blkz)) = ...
        noise_gp(blkstartx(blkx):blkendx(blkx), blkstarty(blky):blkendy(blky),...
        blkstartz(blkz):blkendz(blkz)) + blk_noise_gp;
    
    if(mean(abs(blk.yint(:)+blk.yext(:)))>3*mean(abs(ytr_blk(:))))
        error('bad block')
    end
    
    
end

%%
clear ytr

%load([datafolder,filename],'mask_wholebrain')

counter(counter==0)=1;
repmatcounter = repmat(counter,[1,1,1,sz(4)]);


for fi = 1:length(nii_filenames)

    nii_filename = nii_filenames{fi};
    try
        nii = load_nii(fullfile(niidatafolder,[nii_filename,'.nii']));
    catch err
        nii = load_untouch_nii(fullfile(niidatafolder,[nii_filename,'.nii']));
    end

       % divide by repmatcounter to average overlapping areas;
    nii.img(mask_wholebrain{1},mask_wholebrain{2},mask_wholebrain{3},:) = squeeze(ytr_gp(:,:,:,:,fi)./repmatcounter)+BIAS;
    nii.img(isnan(nii.img))=0;

    try
        save_nii(nii, fullfile(niisavefolder,[nii_filename,'_GP.nii']))

    catch err
        save_untouch_nii(nii, fullfile(niisavefolder,[nii_filename,'_GP.nii']))

    end
end


% nii_filename = nii_filenames{1};
% try
%     nii = load_nii(fullfile(niidatafolder,[nii_filename,'.nii']));
% catch err
%     nii = load_untouch_nii(fullfile(niidatafolder,[nii_filename,'.nii']));
% end
% 
% nii.img = zeros(size(nii.img(:,:,:,1)));
% nii.img(mask_wholebrain{1},mask_wholebrain{2},mask_wholebrain{3}) = squeeze(noise_gp./counter);
% nii.img(isnan(nii.img))=0;
% 
% try
%     save_nii(nii, fullfile(niisavefolder,[nii_filename,'_noiseSTDmap.nii']))
%     
% catch err
%     save_untouch_nii(nii, fullfile(niisavefolder,[nii_filename,'_noiseSTDmap.nii']))
%     
% end
