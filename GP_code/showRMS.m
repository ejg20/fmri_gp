function trendrms = showRMS(Y, T1file, view, sz, lims, mask_wholebrain, mask)

clf
numofrun = size(Y,3);
trendrms = zeros(prod(sz(1:3)),numofrun);

if(nargin < 5 || isempty(lims))
    lims = [1,sz(view)];
end

if(nargin < 7 || isempty(mask))
    mask=ones(sz(1:3));
end

for i=1:numofrun
    trendrms(:,i) = std(squeeze(Y(:,:,i)),[],2);
end

mintrend = min(trendrms(:));
maxtrend = max(trendrms(:))/4;
% for frame = 1:sz(view);
%     for i=1:numofrun
%         subplot(2,2,i);
%         im3d = reshape(trendrms(:,i),sz(1:3));
%         switch view
%             case 1
%                 imagesc(squeeze((im3d(frame,:,:))));
%             case 2
%                 imagesc(squeeze((im3d(:,frame,:))));
%             case 3
%                 imagesc(rot90(im3d(:,:,frame)));
%             otherwise
%                 error('Wrong view');
%         end
%         caxis([mintrend,maxtrend]);
%     end
%     pause;
% end

T1 = load_nii(T1file);
T1.img = T1.img(mask_wholebrain{1},mask_wholebrain{2},mask_wholebrain{3}).*mask;

numofframes=lims(2)-lims(1)+1;
if(numofframes == 1)
    view = [1,2,3];
    numofframes=3;
    frames = lims(1)*ones(3,1);
else
    view = repmat(view,numofframes,1);
    frames = (1:numofframes)+lims(1)-1;
end
spacing=0.001; padding=0.001; margin=0.1;
for run = 1:numofrun+1
    if(run>1)
    	im3d = reshape(trendrms(:,run-1),sz(1:3)).*mask;
    end
    for i = 1:numofframes;
        fi = frames(i);
        subaxis(numofrun+1,numofframes,i+(run-1)*numofframes, 'Spacing', spacing, 'Padding', padding, 'Margin', margin);
        switch view(i)
            case 1
                if(run == 1)
                    imagesc(rot90(squeeze(T1.img(fi,:,:))));colormap(gray); 
                else
                    imagesc(rot90(squeeze(im3d(fi,:,:))));colormap(jet)
                    caxis([mintrend,maxtrend]);
                end
                
            case 2
                if(run == 1)
                    imagesc(rot90(squeeze(T1.img(:,fi,:))));colormap(gray);
                else
                    imagesc(rot90(squeeze(im3d(:,fi,:))));colormap(jet)
                    caxis([mintrend,maxtrend]);
                end
            case 3
                if(run == 1)
                    imagesc(rot90(T1.img(:,:,fi)));colormap(gray); 
                else
                    imagesc(rot90(im3d(:,:,fi)));colormap(jet)
                    caxis([mintrend,maxtrend]);
                end
            otherwise
                error('Wrong view');
        end
        freezeColors
        set(gca,'XTick',[]), set(gca,'YTick',[]);
    end
end


end