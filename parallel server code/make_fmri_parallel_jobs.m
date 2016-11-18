function [] = make_fmri_parallel_jobs(datafilename, numofparalleljobs, datafolder, savefolder)

if(nargin<1)
    % datafilename = '88_100316CR_1-pol_20140129T132253';
    datafilename = 'fmriGP_whole_20140130T111128';
end

if(nargin<2)
    numofparalleljobs = 50;
end

filename = 'fmriGP_script_wholebrain_p';
 
fidbatch = fopen(fullfile(savefolder,'batch_run'), 'w');
fprintf(fidbatch, '#!/bin/sh\n');
fprintf(fidbatch, '#$ -cwd\n');


for i =1:numofparalleljobs
    
    fidmfile = fopen(fullfile(savefolder,[filename,num2str(i),'.m']), 'w');
    fprintf(fidmfile,'addpath(genpath(%s))\n','''/home/research/gilboae/fMRI''');
    fprintf(fidmfile,'addpath(genpath(%s))\n','''/home/research/gilboae/gp''');
    fprintf(fidmfile, 'run_fmri_parallel_jobs(%d, %s%s%s, %d)\n',i,'''',datafilename,'''',numofparalleljobs);
    fclose(fidmfile);
    
    fidshell = fopen(fullfile(savefolder,['job',num2str(i),'.sh']), 'w');
    fprintf(fidshell, '#!/bin/sh\n');
    fprintf(fidshell, '#$ -cwd\n');
    fprintf(fidshell, ['/cluster/cloud/matlab/bin/matlab -nodisplay -nojvm <', filename,num2str(i),...
        '.m \n']);
    fclose (fidshell);
   
    fprintf(fidbatch, ['qsub job%d.sh','\n'],i);
    
end

fclose(fidbatch);