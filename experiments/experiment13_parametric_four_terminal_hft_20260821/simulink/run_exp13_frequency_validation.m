function run_exp13_frequency_validation()
%RUN_EXP13_FREQUENCY_VALIDATION Compare Simscape Y4 with independent MNA.

root=fileparts(mfilename('fullpath')); old=pwd; c=onCleanup(@()cd(old)); cd(root);
res=fullfile(root,'..','results'); if ~exist(res,'dir'),mkdir(res);end
cases={make_exp13_parameters(4,4,'regression4'),make_exp13_parameters(8,8,'candidate8')};
f=logspace(3,7,121).'; summary=cell(2,7);
for m=1:2
    p=cases{m}; name=sprintf('hft_four_terminal_%dp_%ds_%s',p.Np,p.Ns,p.name);
    file=fullfile(root,'generated',[name '.slx']);
    if ~exist(file,'file'),file=build_parametric_four_terminal_hft(p,name);end
    load_system('hftparam_lib');load_system(file);
    [A,B,C,D]=linmod(name);
    Ysim=zeros(4,4,numel(f));Yref=Ysim;rel=zeros(numel(f),1);
    for k=1:numel(f)
        s=1i*2*pi*f(k);Ysim(:,:,k)=C*((s*eye(size(A))-A)\B)+D;
        Yref(:,:,k)=reference_y4(f(k),p);
        rel(k)=norm(Ysim(:,:,k)-Yref(:,:,k),'fro')/max(norm(Yref(:,:,k),'fro'),eps);
    end
    recip=max(arrayfun(@(k)norm(Ysim(:,:,k)-Ysim(:,:,k).','fro'),1:numel(f)));
    summary(m,:)={name,size(A,1),mean(rel),max(rel),recip,min(eig(p.L)),nnz(p.Cps)};
    save(fullfile(res,[name '_linearization.mat']),'A','B','C','D','f','Ysim','Yref','rel','p');
    T=table(f,rel,'VariableNames',{'frequency_hz','fro_relative_error'});
    writetable(T,fullfile(res,[name '_frequency_error.csv']));
    close_system(name,0);
end
S=cell2table(summary,'VariableNames',{'model','state_count','mean_fro_error','max_fro_error','max_reciprocity_residual','lambda_min_L_H','nnz_Cps'});
writetable(S,fullfile(res,'frequency_validation_summary.csv'));disp(S);
end

function Yport=reference_y4(freq,p)
s=1i*2*pi*freq;np=p.Np;ns=p.Ns;nb=np+ns;nn=np+1+ns+1;
G=zeros(nn,nb);
for k=1:np,G(k,k)=1;G(k+1,k)=-1;end
off=np+1;
for k=1:ns,G(off+k, np+k)=1;G(off+k+1,np+k)=-1;end
Yn=G*((diag(p.R)+s*p.L)\G.');
for k=1:nb
    a=G(:,k);Yn=Yn+s*p.Cseries(k)*(a*a.');
end
for k=1:np+1,Yn(k,k)=Yn(k,k)+s*p.Cg_p(k);end
for k=1:ns+1,Yn(off+k,off+k)=Yn(off+k,off+k)+s*p.Cg_s(k);end
for i=1:np
    for j=1:ns
        if p.Cps(i,j)==0,continue,end
        a=zeros(nn,1);a(i)=1;a(off+j)=-1;Yn=Yn+s*p.Cps(i,j)*(a*a.');
    end
end
ports=[1,np+1,off+1,off+ns+1];internal=setdiff(1:nn,ports);
Yport=Yn(ports,ports)-Yn(ports,internal)*(Yn(internal,internal)\Yn(internal,ports));
end

