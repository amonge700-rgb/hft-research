function run_exp14_build_compile()
root=fileparts(mfilename('fullpath'));old=pwd;c=onCleanup(@()cd(old));cd(root);warning('off','all');modelFile=build_exp14_dab_model();[~,name]=fileparts(modelFile);load_system(modelFile);t=tic;set_param(name,'SimulationCommand','update');compile_s=toc(t);close_system(name,0);res=fullfile(root,'..','results');if ~exist(res,'dir'),mkdir(res);end;T=table(string(name),true,compile_s,'VariableNames',{'model','compile_ok','compile_time_s'});writetable(T,fullfile(res,'build_summary.csv'));fprintf('EXP14_COMPILE_OK %s %.6f s\n',name,compile_s);
end
