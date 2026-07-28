function validate_model_compile()
%VALIDATE_MODEL_COMPILE Compile the saved Simscape network and save status.

warning('off','all');
root = fileparts(mfilename('fullpath'));
cd(root);
load_system('hftlib_lib');
mdl = 'hft_segmented_ladder_simscape';
load_system(fullfile(root,[mdl '.slx']));

statusFile = fullfile(root,'simscape_compile_status.txt');
fid = fopen(statusFile,'w');
cleanup = onCleanup(@() fclose(fid));
try
    set_param(mdl,'SimulationCommand','update');
    fprintf(fid,'PASS\n');
    fprintf(fid,'Model compiled successfully in MATLAB/Simulink R2021b.\n');
    fprintf(fid,'Timestamp: %s\n',datestr(now,30));
    fprintf('SIMSCAPE_COMPILE_PASS\n');
catch ME
    fprintf(fid,'FAIL\n%s\n',getReport(ME,'extended','hyperlinks','off'));
    fprintf('SIMSCAPE_COMPILE_FAIL\n%s\n',ME.message);
    rethrow(ME);
end
close_system(mdl,0);
end
