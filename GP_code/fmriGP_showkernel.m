function [] = fmriGP_showkernel(ytr, hypers_learned, gp_input, gpmodel, varargin)


pnames = {'lims','orientation','labels'};
dflts =  {[],0,{'Run','Time','Z','Y','X'};};
[lims, orientation,labels] = internal.stats.parseArgs(pnames, dflts, varargin{:});

lblsz = length(labels);
% ylabels = {'Run','Time','Z','Y','X'};
P = gp_input.get_P;
for p = 1:P
    if(p>=P-2 && P>3)
        sz = size(ytr)';
        endpoint = max((1./gp_input.Fs(P-2:P)).*sz(1:3));
    else
        endpoint =1/gp_input.Fs(p)*size(ytr,P+1-p);
    end
    xseries = (0:1/gp_input.Fs(p)/500:endpoint)';
    covfunc = gpmodel.cov{1}{p}; hyp.cov = hypers_learned(gpmodel.hyps_in_d{1}{p});
    K1_rec = feval(covfunc{:}, hyp.cov, 0,xseries);
    if(orientation == 0)
        subplot(P,1,P-p+1); plot(xseries,K1_rec(1,:)/K1_rec(1),'linewidth',3); 
        ylabel('Correlations','fontsize',14);
    else
        subplot(1,P,P-p+1); plot(xseries,K1_rec(1,:)/K1_rec(1),'linewidth',3); 
        if p == P;
            ylabel('Correlations','fontsize',14);
        end
    end
        %      subplot(P,1,P-p+1); plot(xseries,K1_rec(1,:)); 
  
    xlabel(labels{lblsz-P+p},'fontsize',14);
    set(gca,'fontsize',14)
    
    axis tight
    if(isempty(lims))
        xlimend = find(abs(K1_rec(1,:)/K1_rec(1))>1e-6,1,'last');
        if(~isempty(xlimend))
            xlim([0,xseries(xlimend)])
        end
    else
         xlim(lims(P-p+1,:))
    end
    
end
