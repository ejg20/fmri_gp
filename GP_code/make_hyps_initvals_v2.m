function hyps_init = make_hyps_initvals_v2(input, gpmodel, initmethod, paramsvec)

Z = length(gpmodel.hyps_in_d);
P = length(gpmodel.hyps_in_d{1});


hyps_init = zeros(gp_grid_numofhyps_v2(gpmodel.hyps_in_d),1);
 
for z = 1:Z
    params = paramsvec(z);
    for p = 1:P
        if(length(input.Fs) == P)
            Fs = input.Fs(p);
        else
            Fs = input.Fs(1);
        end

        switch(gpmodel.cov{z}{p}{1})
            case 'covSM1D'
                switch initmethod
                    case  1
                        for p=1:P
                            Q = length(gpmodel.hyps_in_d{p})/3;
                            hyps_init(gpmodel.hyps_in_d{p}(1:Q)) = randn(Q,1)*params.ws+params.wm;
                            hyps_init(gpmodel.hyps_in_d{p}(Q+(1:Q))) =  mod(randn(Q,1)*params.mus+params.mum,log(input.Fs));
                            hyps_init(gpmodel.hyps_in_d{p}(2*Q+(1:Q))) = randn(Q,1)*params.ss+params.sm;
                        end
                    case  2
                        for p=1:P
                            Q = length(gpmodel.hyps_in_d{p})/3;
                            a = params.wm*sqrt(2*pi)/2;
                            hyps_init(gpmodel.hyps_in_d{p}(1:Q)) = abs(a*randn(Q,1));
                            %         hyps_init(gpmodel.hyps_in_d{p}(q+(1:q))) =  mod(mum+rand(q,1)*mus,Fs/2);
                            %         a = mus*sqrt(2*pi)/2;
                            %         hyps_init(gpmodel.hyps_in_d{p}(q+(1:q))) = abs(a*randn(q,1));
                            hyps_init(gpmodel.hyps_in_d{p}(Q+(1:Q))) = gamrnd(params.muA,params.muB/10,Q,1);
                            a = params.sm*sqrt(2*pi)/2;
                            hyps_init(gpmodel.hyps_in_d{p}(2*Q+(1:Q))) = 1./(abs(a*randn(Q,1)));
                        end
                        hyps_init = log(hyps_init);
                        
                    case 3
                        Q = gpmodel.cov{z}{p}{2};
                        
                        %%%%% DETERMINISTIC WEIGHTS (FRACTION OF VARIANCE)
                        w0 = ((params.wm/Z)^(1/P)/Q*sqrt(2*pi)/2)*ones(Q,1);
                        hyps_init(gpmodel.hyps_in_d{z}{p}(1:Q)) = w0;
                        
                        %%%%% UNIFORMLY RANDOM FREQS
%                         mu = max(Fs/2*rand(q,1),1e-8);
                        if(Q == 1)
                            mu = 1e-10;
                        else
                            mu = abs(Fs/2/4*randn(Q,1));
                        end
                        hyps_init(gpmodel.hyps_in_d{z}{p}(Q+(1:Q))) = mu;
                        
                        %%%%%% TRUNCATED GAUSSIAN FOR LENGTHSCALES (1/Sigma)
                        sigmean = params.sm(p)*sqrt(2*pi)/2;
                        hyps_init(gpmodel.hyps_in_d{z}{p}(2*Q+(1:Q))) = 1./(abs(sigmean*randn(Q,1)));
%                         sigmean = params.sm(p);
%                         hyps_init(gpmodel.hyps_in_d{z}{p}(2*Q+(1:Q))) = 1./(abs(sigmean*ones(Q,1)));

                end
                
            case {'covSEard','covMaterniso'}
                a = sqrt(params.sm(p)*sqrt(pi/2));
                hyps_init(gpmodel.hyps_in_d{z}{p}) = [abs(a*randn);(params.wm/Z)^(1/P)];
            case 'covRQard'
                a = sqrt(params.sm(p)*sqrt(pi/2));
                hyps_init(gpmodel.hyps_in_d{z}{p}) = [abs(a*randn);(params.wm/Z)^(1/P);1];
        end
    end
end
    hyps_init = log(hyps_init);
end

function projSpect = project_spectrum_to_d(spect,d)
D = length(size(spect));
tempProj = spect;

for i = D:-1:1;
    if(i~=d)
        tempProj = squeeze(mean(tempProj,i));
    end
end
projSpect = tempProj(:);

end