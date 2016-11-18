%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% showSMkernel()
%
% Plot the spectrum of the SM kernel from the hyperparameters.
%
%
% Usage: [CAX] = showSMkernel(hypvec, hyps_in_dim, Fs, varargin)
%
% Inputs:
%     hypvec        hyperparameters vector
%     hyps_in_dim   A cell vector [size D]. Each cell contains a vector of
%                   the hyperparameters in the corresponding dimension. The
%                   parameters must be in the order to be used in the
%                   covariance function.
%     Fs            Sampling frequency
%
% Outputs:
%     CAX           Color axis for comparison with other images
%
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [CAX] = showSMkernel_v2(hypvec, hyps_in_d, Fs, varargin)

pnames = {   'CAX' 'titlestr' 'num_of_spec_points' 'axislimit'};
dflts =  {[], 'Log SM Kernel Spectrum', 1000, [] };
[CAX titlestr num_of_spec_points axislimit] ...
    = internal.stats.parseArgs(pnames, dflts, varargin{:});

Z = length(hyps_in_d);

if(nargin < 1)
    error('no hyper parameters vector');
elseif(nargin < 2)
    error('no hyps_in_dim');
elseif(nargin < 3)
    error('no sampling frequency');
end


% fx = linspace(0,1,num_of_spec_points+1);
% fx = Fs/2*[-fliplr(fx),fx(2:end)];
% fy = linspace(0,1,num_of_spec_points+1);
% fy = Fs/2*[-fliplr(fy),fy(2:end)];

fx = Fs/2*linspace(0,1,num_of_spec_points+1);
fy = Fs/2*linspace(0,1,num_of_spec_points+1);


fftxgrid{1} = fx;
fftxgrid{2} = fy;

specdensityImage = 0;
for z = 1:Z
    hyps_in_d_z = hyps_in_d{z};
    specdensityImage_z = multdimSpecdensity(fftxgrid,hypvec(:),hyps_in_d_z);
    specdensityImage = specdensityImage+specdensityImage_z;
end

[X,Y] = meshgrid(fx,fy);
axes('FontSize',24);
% surf(X,Y,log(abs(specdensityImage) + 1),'EdgeColor','none')
imagesc(log(abs(specdensityImage) + 1)); %colormap(gray)
% pcolor(X,Y,log(abs(specdensityImage) + 1)); colormap(gray)
view(2)
% colormap(gray);
xlabel('Hz','FontSize',24);

% if(exist('titlestr','var') && ~isempty(titlestr) && ischar(titlestr))
%     title(titlestr);
% end

if(exist('axislimit','var') && ~isempty(axislimit) )
    axis(axislimit);
end

if(~isempty(CAX))
    caxis(CAX)
else
    CAX = caxis;
end
% colorbar('FontSize',24);
