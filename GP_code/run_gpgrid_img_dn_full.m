function [gpimg] = run_gpgrid_img_dn_full(filename,datafolder,testfolder,savefolder,edges,ineq,thresh,closeholes,angle,subdir,maskfilename)


% %%%%%%%% HORSE
% I_big=load('aug304_0');
% dir_name = 'horse';
% frame0=I_big.aug25_0(480:580,780:880);
% data = frame0;
% [mask] = (frame0>800);%makemask(frame0.*(frame0<800), 0.6,2000,5);

% ============================= PCB ===========================\
% filename = 'sphere_0';
% load(['C:\Users\Elad\Documents\Research\phaseCam\big data\',filename,'.mat'])
% frame0 = data(:,:,1);
% dir_name = filename;
% %
% % img = frame0(90:190,50:150);     %object
%
% img = frame0(1:200,1:200);
% data = data(1:200,1:200,:);
% mask=zeros(size(img));
% mask = img<300;

% ============================= Bottle ===========================\

loadmatfile = matfile([datafolder,filename,'.mat']);


frame0 = loadmatfile.data(:,:,1);
data = loadmatfile.data(edges(1):edges(2),edges(3):edges(4),:);
dir_name = filename;

img = frame0(edges(1):edges(2),edges(3):edges(4),1);     %object
% data = data(181:340,241:400,1:900);

mask=zeros(size(img));
% mask = img > 1500;          %text
if(ineq < 0)
    mask = img < thresh;            %background
elseif(ineq > 0)
    mask = img > thresh;            %object
end


if(closeholes)
    mask = bwareaopen(mask,1000);
    mask = 1-bwareaopen(1-mask,1000);
end

if(~isempty(maskfilename))
    load(maskfilename);
    if(ineq < 0)
        mask = 1-mask;
    end
end

xsize = size(img,1);
ysize = size(img,2);

if(angle == 0)
    xsubindexstart = 1;
    ysubindexstart = 1;
elseif(angle == 45)
    xsubindexstart = 2;
    ysubindexstart = 1;
elseif(angle == 90)
    xsubindexstart = 1;
    ysubindexstart = 2;
elseif(angle == 135)
    xsubindexstart =2;
    ysubindexstart =2;
end
xsubindex = xsubindexstart:xsize;
ysubindex = ysubindexstart:ysize;

sub_img = img(xsubindex,ysubindex);
mv = mean(data,3);
vv = var(data,[],3);
pfit=polyfit(mv(:),vv(:),1);
p = [0.2495   15.9858];

% % maskIndx = find(mask == 0);
% % p=polyfit(mv(maskIndx),vv(maskIndx),1);
esn = (p(1)*sub_img+p(2));
if(min(esn(:))<=0)
    %     y = (p(1)*max(sub_img(:))+p(2));
    %     p(1) = (y-min(vv(:)))/(max(mv(:))-min(mv(:)));
    %     p(2) = min(vv(:));
    
    p(2) = p(2)+min(esn);
    
    % %     y = (p(1)*max(sub_img(maskIndx))+p(2));
    % %     p(1) = (y-min(vv(maskIndx)))/(max(mv(maskIndx))-min(mv(maskIndx)));
    % %     p(2) = min(vv(maskIndx));
    % %     esn = (p(1)*sub_img+p(2));
    %     esn = (y-min(vv(:)))/(max(mv(:))-min(mv(:)))*sub_img+min(vv(:));
    %     (p(1)*sub_img+p(2))-min(esn(:))+min(vv(:));       %correction to noise model so will not get negative vlues
end
esn = (p(1)*sub_img+p(2));

sn = zeros(size(sub_img));
imagesc(mask);
mask_sub = mask(1:xsize,1:ysize);
sn(logical(mask_sub)) = 1500*max(sub_img(:));%0*max(max(B.*(1-mask_sub)));
sn = sn+esn;


% h = fspecial('average', 3);
% edgnnoisemask = 1-(min(filter2(h, 1-mask_sub),1-mask_sub));
% imagesc(edgnnoisemask);
% sn = (1+2*edgnnoisemask).*sn;


Xgrid = cell(2,1);
Xgrid{2} = xsubindex;
Xgrid{1} = ysubindex;

startTime = rem(now,1);
[gpimg, logtheta_learned_dn,Stdfull] = gpgrid_img_dn(sub_img, Xgrid, xsize, ysize, sn(:), mask);
% [gpimg, logtheta_learned_dn] = gpgrid_img(sub_img,Xgrid, xsize, ysize);
exec_time = (rem(now,1)-startTime)*24*3600;

%%
% close all;


while(isempty(dir_name))
    dir_name = input('Directory name: ', 's');
end

% while(1)
%     svfolder = ['C:/Users/Elad/Documents/Research/phaseCam/tests/',dir_name,'/'];
%     mkdir(svfolder)
%     if (isequal(lastwarn,'Directory already exists.'))
%         reply = input('Directory already exists. Do you want to continue? Y/N [Y]: ', 's');
%         if ~isequal(reply,'Y')
%             reply = 'Y';
%             break;
%         end
%     else
%         break;
%     end
% end

%%
fntsz = 30;
svfolder = [savefolder,dir_name,'/',subdir,'/'];
mkdir(svfolder)

copyfile([testfolder,'showFigureseEmpty.tex'],[svfolder,'showFigures.tex'])
copyfile([testfolder,'makepdf.bat'],[svfolder,'makepdf.bat'])
fid = fopen([svfolder,'showFigures.tex'], 'a');


fprintf(fid, '\\section{%s}\n',strrep(dir_name, '_', ' '));

fprintf(fid, 'exec time: %5.2f, \n',exec_time);
fprintf(fid, 'theta= (%3.2f, %3.2f, %3.2f)\\\\\n',exp(logtheta_learned_dn(1)),exp(logtheta_learned_dn(2)),exp(logtheta_learned_dn(3)));
fprintf(fid, 'model p = [0.2495, 15.9858]; \\\\ p = [%5.4f, %5.4f]\\\\\n',p(1),p(2));
fprintf(fid, 'pfit = [%5.4f, %5.4f\n',pfit(1),pfit(2));

imageTrue = mean(data,3);
new_mask = mask;
new_mask(1:2,:)=1;
new_mask(:,2:1)=1;
new_mask(end-1:end,:)=1;
new_mask(:,end-1:end)=1;

num_non_zero = sum(1-new_mask(:));
% new_mask = (imageTrue<160);
% new_mask = (imageTrue>2500);
figIdx = 1;


figure; imshow(mat2gray(imageTrue)); title('True image','FontSize',fntsz);
figureName = 'trueImage';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.25)

figure; imshow(mat2gray(1-new_mask)); title('mask','FontSize',fntsz);
figureName = 'mask';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.25)

figure; imshow(mat2gray(imageTrue.*(1-new_mask))); title('True image (after mask)','FontSize',fntsz);
figureName = 'trueImageAfterMaskGr';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.3)

figure;  imagesc((imageTrue.*(1-new_mask))); title('True image after mask)','FontSize',fntsz);
figureName = 'trueImageAfterMask';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)

figure;plot(mv(:),vv(:),'.'); title('variation vs mean','FontSize',fntsz);
hold on;
plot(sub_img(:),esn(:),'r.')
figureName = 'varVsMean';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)


figureName = 'invKy';figIdx = figIdx+1;
if(sum(findobj('Type','figure') == 11))     %is figure 11 open?
    figure(11); colorbar;
    print( gcf , '-depsc' , [svfolder,figureName ]);
end
writeFigure(fid,[figureName,'.eps'],0.3)

err_MSEgp =  sum(sum(((imageTrue-gpimg).^2).*(1-new_mask)))/num_non_zero
err_SMSEgp =  sum(sum(((imageTrue-gpimg).^2)./(p(1)*imageTrue+p(2)).*(1-new_mask)))/num_non_zero
xlabeltext={'MSE GP = ',num2str(err_MSEgp);'SMSE GP = ',num2str(err_SMSEgp)};
figure; imshow(mat2gray(gpimg.*(1-new_mask)));title('GP interpolation','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'GPimgGr';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.3)

figure; imagesc((gpimg.*(1-new_mask)));title('GP interpolation','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'GPimg';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)


I_sub = sub_img;
I_bcsp = abs(cbsi (I_sub,1));
I_bcsp = zeropadimage(I_bcsp,imageTrue,xsubindexstart,ysubindexstart);
err_MSEbicsp =  sum(sum(((imageTrue-I_bcsp).^2).*(1-new_mask)))/num_non_zero
err_SMSEbicsp =  sum(sum(((imageTrue-I_bcsp).^2)./(p(1)*imageTrue+p(2)).*(1-new_mask)))/num_non_zero
xlabeltext = {'MSE BICSP = ',num2str(err_MSEbicsp);'SMSE BICSP = ',num2str(err_SMSEbicsp)};
figure; imshow(mat2gray(I_bcsp.*(1-new_mask)));title('Bicubic-Spline interpolation','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'BICSPimgGr';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.3)

figure; imagesc((I_bcsp.*(1-new_mask)));title('Bicubic-Spline interpolation','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'BICSPimg';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)

I_sub = sub_img(1:1:end,1:1:end);
I_bcsp1 = abs(cbsi (I_sub,1));
I_bcsp1 = zeropadimage(I_bcsp1,imageTrue,xsubindexstart,ysubindexstart);
h = fspecial('average', 3);
I_bcsp1 = filter2(h, I_bcsp1);
err_MSEbicsp_lp1 =  sum(sum(((imageTrue-I_bcsp1).^2).*(1-new_mask)))/num_non_zero
err_SMSEbicsp_lp1 =  sum(sum(((imageTrue-I_bcsp1).^2)./(p(1)*imageTrue+p(2)).*(1-new_mask)))/num_non_zero
xlabeltext = {'MSE BICSP = ',num2str(err_MSEbicsp_lp1);'SMSE BICSP = ',num2str(err_SMSEbicsp_lp1)};
figure; imshow(mat2gray(I_bcsp1.*(1-new_mask)));title('Bicubic-Spline interpolation (low pass after interpolation)','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'BICSPimgLPassAfterGr';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.3)

figure; imagesc((I_bcsp1.*(1-new_mask)));title('Bicubic-Spline interpolation (low pass after interpolation)','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'BICSPimgLPassAfter';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)

h = fspecial('average', 3);
BSmoothed = filter2(h, sub_img);
I_sub = BSmoothed(1:1:end,1:1:end);
I_bcsp2 = abs(cbsi(I_sub,1));
I_bcsp2 = zeropadimage(I_bcsp2,imageTrue,xsubindexstart,ysubindexstart);
err_MSEbicsp_lp2 =  sum(sum(((imageTrue-I_bcsp2).^2).*(1-new_mask)))/num_non_zero
err_SMSEbicsp_lp2 =  sum(sum(((imageTrue-I_bcsp2).^2)./(p(1)*imageTrue+p(2)).*(1-new_mask)))/num_non_zero
xlabeltext = {'MSE BICSP = ',num2str(err_MSEbicsp_lp2);'SMSE BICSP = ',num2str(err_SMSEbicsp_lp2)};
figure; imshow(mat2gray(I_bcsp2.*(1-new_mask)));title('smooth Bicubic-Spline interpolation (low pass before interpolation)','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'BICSPimgLPassBeforerGr';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.3)

figure; imagesc((I_bcsp2.*(1-new_mask)));title('smooth Bicubic-Spline interpolation (low pass before interpolation)','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'BICSPimgLPassBeforer';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)


I_sub = sub_img(1:1:end,1:1:end);
I_nedi = abs(sri(I_sub,1));
I_nedi = zeropadimage(I_nedi,imageTrue,xsubindexstart,ysubindexstart);
err_MSEnedi =  sum(sum(((imageTrue-I_nedi).^2).*(1-new_mask)))/num_non_zero
err_SMSEnedi =  sum(sum(((imageTrue-I_nedi).^2)./(p(1)*imageTrue+p(2)).*(1-new_mask)))/num_non_zero
xlabeltext = {'MSE NEDI = ',num2str(err_MSEnedi);'SMSE NEDI = ',num2str(err_SMSEnedi)};
figure; imshow(mat2gray(I_nedi.*(1-new_mask)));title('New Edge-Directed Interpolation','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'NEDIimgGr';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.3)

figure; imagesc((I_nedi.*(1-new_mask)));title('New Edge-Directed Interpolation','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'NEDIimg';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)


I_sub = sub_img(1:1:end,1:1:end);
I_bc = abs(cbi_modi_0(I_sub,size(imageTrue,1),size(imageTrue,2)));
I_bc = zeropadimage(I_bc,imageTrue,xsubindexstart,ysubindexstart);
err_MSEbc =  sum(sum(((imageTrue-I_bc).^2).*(1-new_mask)))/num_non_zero
err_SMSEbc =  sum(sum(((imageTrue-I_bc).^2)./(p(1)*imageTrue+p(2)).*(1-new_mask)))/num_non_zero
xlabeltext = {'MSE Bicubic = ',num2str(err_MSEbc);'SMSE Bicubic = ',num2str(err_SMSEbc)};
figure; imshow(mat2gray(I_bc.*(1-new_mask)));title('Bicubic Interpolation','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'BICimgGr';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.3)

figure; imagesc((I_bc.*(1-new_mask)));title('Bicubic Interpolation','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'BICimg';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)


I_sub = sub_img(1:1:end,1:1:end);
I_blin = abs(bilinear_bobcat1k(I_sub));
I_blin = zeropadimage(I_blin,imageTrue,xsubindexstart,ysubindexstart);
err_MSEblin =  sum(sum(((imageTrue-I_blin).^2).*(1-new_mask)))/num_non_zero
err_SMSEblin =  sum(sum(((imageTrue-I_blin).^2)./(p(1)*imageTrue+p(2)).*(1-new_mask)))/num_non_zero
xlabeltext = {'MSE Blinear = ',num2str(err_MSEblin);'SMSE Blinear = ',num2str(err_SMSEblin)};
figure; imshow(mat2gray(I_blin.*(1-new_mask)));title('Bilinear Interpolation','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'BIlinimgGr';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.3)

figure; imagesc((I_blin.*(1-new_mask)));title('Bilinear Interpolation','FontSize',fntsz); xlabel(xlabeltext,'FontSize',fntsz);
figureName = 'BIlinimg';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)

save([svfolder,'workspace'])

fprintf(fid, '\\end{document}\n');

fclose(fid);

oldFolder = cd(svfolder);
system('copy "showFigures.tex" "showFigures1.tex"');
system('latex.exe --src "showFigures1.tex"');
% system('dvipdfmx.exe "showFigures1.dvi"');
% pause(1)
% system('"C:\Program Files (x86)\Adobe\Reader 10.0\Reader\AcroRd32.exe" showFigures1.pdf');
cd(oldFolder)

close all;
end
