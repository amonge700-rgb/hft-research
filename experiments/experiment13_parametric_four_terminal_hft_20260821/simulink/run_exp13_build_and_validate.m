function run_exp13_build_and_validate()
%RUN_EXP13_BUILD_AND_VALIDATE Build 4-section regression and 8-section models.

root=fileparts(mfilename('fullpath')); old=pwd; c=onCleanup(@()cd(old)); cd(root);
res=fullfile(root,'..','results'); if ~exist(res,'dir'),mkdir(res);end
diary(fullfile(res,'build_and_compile_log.txt')); cleanupDiary=onCleanup(@()diary('off'));

cases={make_exp13_parameters(4,4,'regression4'),make_exp13_parameters(8,8,'candidate8')};
rows=cell(numel(cases),8);
for m=1:numel(cases)
    p=cases{m}; name=sprintf('hft_four_terminal_%dp_%ds_%s',p.Np,p.Ns,p.name);
    fprintf('\n=== Building %s ===\n',name);
    file=build_parametric_four_terminal_hft(p,name);
    load_system(file);
    compileOK=false; msg=''; tic;
    try
        set_param(name,'SimulationCommand','update'); compileOK=true;
    catch ME
        msg=getReport(ME,'extended','hyperlinks','off'); fprintf('%s\n',msg);
    end
    elapsed=toc; close_system(name,0);
    rows(m,:)={name,p.Np,p.Ns,p.Np+p.Ns,nnz(p.Cps),min(eig(p.L)),compileOK,elapsed};
end
T=cell2table(rows,'VariableNames',{'model','Np','Ns','Nb','nnz_Cps','lambda_min_L_H','compile_ok','compile_time_s'});
writetable(T,fullfile(res,'build_validation_summary.csv'));
save(fullfile(res,'exp13_parameters.mat'),'cases','T');
disp(T);
end

