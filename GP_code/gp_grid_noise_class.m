classdef gp_grid_noise_class < handle
    % The following properties can be set only by class methods
    properties (SetAccess = private)
    end
    properties (SetAccess = public)
        var
        learn
        sphericalNoise
        groups
    end
    methods
        function gp_noise = gp_grid_noise_class(noisevar, indx)
            gp_noise.setvar(noisevar, indx);
            gp_noise.groups = {indx};
        end
        function gp_noise = setvar(gp_noise, noisevar, indx)
            gp_noise.var = noisevar;
            if(nargin < 3 || numel(indx)~=numel(gp_noise.var) || sum(noisevar(:))~=numel(gp_noise.var) )
                gp_noise.sphericalNoise = false;
            else
                gp_noise.sphericalNoise = true;
            end
            
        end
        
    end % methods
end % classdef
