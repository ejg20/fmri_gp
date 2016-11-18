classdef gp_grid_input_class < handle
%     input         Input structure for GP-grid
%       .index_to_N	Indices of not dummy locations
%       .data       The sub image to be interpolated
%       .xgrid         Cell array of per dimension grid points
%     noise_struct
%       .learn      flag if to learn noise hyperparameter
%       .var        noise variance of observations. If is a vector and
%                   noise_struct.learn == 1, then will only learn the noise
%                   for not dummy locations. if noise_struct.learn == 0,
%                   then noise_struct.var contains the known noise
%                   variance.
%       .sphericalNoise
%                   learn a single noise hyperparameter for entire input
%                   space (spherical noise).
%     mask          Binary mask for output image
%     InitParamSet  Values of initial parameters.
%       .learn      If true, then use initial values to find optimal hypers.
%                   If false, then use the first row of .vals as the
%                   optimal parameters (no learning).
%       .vals       Matrix where each row
%                   is an initial guess for all hypers. If learning noise
%                   hyperparameter, then the noise hyperparameter must be
%                   located a the end of the hyperparameter vals vector
%     cov           Covariance function as in gpml-matlab
%     hyps_in_dim   A cell vector [size D]. Each cell contains a vector of
%                   the hyperparameters in the corresponding dimension. The
%                   parameters must be in the order to be used in the
%                   covariance function.
    
    
    
    % The following properties can be set only by class methods
    properties (SetAccess = private)
        
    end
    properties (SetAccess = public)
        xgrid;
        index_to_N;
        zeromeandata;
        meandata;
        Fs;
        P;
    end
    % Define an event called InsufficientFunds
    methods
        function gp_input = gp_grid_input_class(varargin)
            
            if(isa(varargin{1},'gp_grid_input_class'))
                 % function signature is gp_grid_input_class(gp_input)
                gpinput_copyfrom = varargin{1};
                gp_input.xgrid=gpinput_copyfrom.xgrid;
                gp_input.P=gpinput_copyfrom.P;
                gp_input.index_to_N=gpinput_copyfrom.index_to_N;
                gp_input.zeromeandata=gpinput_copyfrom.zeromeandata;
                gp_input.meandata=gpinput_copyfrom.meandata;
                gp_input.Fs=gpinput_copyfrom.Fs;
            else
                % function signature is gp_grid_input_class(data, index_to_N, Fs, Gs)
                data = varargin{1};     %data contains all the N input elements of the grid
                gp_input.index_to_N = varargin{2}(:); % index_to_N containt the indices of the n elements that are not masked out
                gp_input.meandata = mean(data(gp_input.index_to_N));
                gp_input.zeromeandata = data(gp_input.index_to_N) - gp_input.meandata;
                if(length(varargin) < 3)
                    gp_input.Fs = 1;
                else
                    gp_input.Fs = flipud(varargin{3}(:));
                end
                if(length(varargin) < 4)    % no Gs (xgrid dimensions) specified
                    szdata = size(data);
                    szindx = find(szdata>1,1,'last');    %find the last index that is bigger then 1. take care of cases 
                                                        %where for example a vectore will be represented as [N,1]
                    Dim = length(szdata(1:szindx));
                    Gs = szdata(1:szindx);
                else
                    Gs = varargin{4};
                    Dim = length(Gs);
                end
                    Vecs = cell(1,Dim);
                    for d = 1:Dim
                        if(length(gp_input.Fs)==Dim)
                            Fs_d = gp_input.Fs(Dim - d + 1);
                        else
                            Fs_d = gp_input.Fs;
                        end
                        Vecs{d} = 1+(0:Gs(d)-1)/Fs_d;
                    end
                    gp_input.make_xgrid(Vecs{:});
                    
                %end
            end
        end
        function copy_xgrid(gp_input, gpinput_copyfrom)
            if(isa(gpinput_copyfrom,'gp_grid_input_class'))
                gp_input.P = gpinput_copyfrom.P;
                gp_input.xgrid = gpinput_copyfrom.xgrid;
            end
        end
        function make_xgrid(gp_input, varargin)
            gp_input.P = length(varargin);
            gp_input.xgrid = fliplr(varargin(:)');
            for d=1:gp_input.P
                gp_input.xgrid{d} = gp_input.xgrid{d}(:);
            end
        end
        function data = get_data(gp_input)
            data = gp_input.zeromeandata(:)+ gp_input.meandata;
        end
        function N = get_N(gp_input)
            N = prod(cellfun(@length, gp_input.xgrid));
        end
        function n = get_n(gp_input)
            n = length(gp_input.index_to_N);
        end
        function P = get_P(gp_input)
            P = gp_input.P;
        end
        function oldmean = set_mean(gp_input, gpmean)
            oldmean = gp_input.meandata;
            gp_input.zeromeandata = gp_input.get_data()  - gpmean;
            gp_input.meandata = gpmean;
            
        end
        function [xstar] = make_xstar(gp_input,varargin)
            P = length(varargin);
            if(P == 1)
                ind_or_subs = varargin{1};
                if(isvector(ind_or_subs))
                    % xstar input is a vector of indices from xgrid
                    possible_xstar = makePossibleComb(gp_input.xgrid);
                    xstar = possible_xstar(ind_or_subs,:);
                else
                    % if xstar is in subs form then just need to flip the matrix in order
                    % to change it from Matlab matrix notation (most rapidly changing dimension
                    % is the first) to gp_grid notation (most rapidly changing dimension is
                    % the last)
                    xstar = fliplr(ind_or_subs);
                end
            else
                % need to build xstar from vectors of subs in each dimensions
                if(length(gp_input.xgrid) ~= P)
                    error('dimension misfit')
                end
                xstar = makePossibleComb(fliplr(varargin));
            end
        end
        function [tr_input, tst_input, cvxstar] = splitsets(gp_input, tstset, addednoise)
            if(nargin < 3)
                addednoise = 0;
            end
            if(isempty(tstset))
                tst_input = gp_grid_input_class(gp_input);
                tr_input = gp_grid_input_class(gp_input);
                tr_input.addgaussiannoise(addednoise);
                cvxstar = gp_input.make_xstar(gp_input.index_to_N);
            else 
                [cvxstar_indx, ~, cvindx_to_N] = intersect(tstset,gp_input.index_to_N);
                cvxstar = gp_input.make_xstar(cvxstar_indx);
                
                Data = zeros(gp_input.get_N(),1);
                Data(gp_input.index_to_N) = gp_input.get_data();
                tst_input = gp_grid_input_class(Data,cvxstar_indx,gp_input.Fs);
                tst_input.copy_xgrid(gp_input);
                
                tr_index_to_N = gp_input.index_to_N;
                tr_index_to_N(cvindx_to_N) = [];   %takeout indices that are in cv
                tr_input = gp_grid_input_class(Data,tr_index_to_N,gp_input.Fs);
                tr_input.copy_xgrid(gp_input);
                tr_input.addgaussiannoise(addednoise);
            end
        end
        function res = isgridcomplete(gp_input)
            res = (gp_input.get_N()==gp_input.get_n());
        end
        function addgaussiannoise(gp_input,addednoise)
            newdata = gp_input.get_data()+addednoise*randn(size(gp_input.index_to_N));
            gp_input.meandata = mean(newdata(:));
            gp_input.zeromeandata = newdata(:)-gp_input.meandata;
        end
    end % methods
end % classdef
