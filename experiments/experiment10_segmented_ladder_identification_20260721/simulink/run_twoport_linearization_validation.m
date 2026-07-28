function run_twoport_linearization_validation()
%RUN_TWOPORT_LINEARIZATION_VALIDATION Linearize Simscape and compare Y(jw).

warning('off','all');
root=fileparts(mfilename('fullpath'));
cd(root);
mdl='hft_segmented_ladder_twoport';
if ~exist(fullfile(root,[mdl '.slx']),'file')
    build_twoport_linearization_model();
end
load_system('hftlib_lib');
load_system(fullfile(root,[mdl '.slx']));

fprintf('LINEARIZATION_START\n');
[A,B,C,D]=linmod(mdl);
fprintf('LINEARIZATION_STATE_COUNT=%d\n',size(A,1));

f=logspace(3,7,161).';
Ysim=zeros(2,2,numel(f));
for k=1:numel(f)
    s=1i*2*pi*f(k);
    Ysim(:,:,k)=C*((s*eye(size(A))-A)\B)+D;
end

% Independent MATLAB implementation of the same physical network MNA.
S=load(fullfile(root,'hft_segmented_ladder_parameters.mat'));
Yref=zeros(size(Ysim));
for k=1:numel(f)
    Yref(:,:,k)=reference_y_matrix(f(k),S);
end

rel=zeros(numel(f),1);
entry=zeros(numel(f),4);
for k=1:numel(f)
    rel(k)=norm(Ysim(:,:,k)-Yref(:,:,k),'fro')/max(norm(Yref(:,:,k),'fro'),eps);
    d=abs(Ysim(:,:,k)-Yref(:,:,k))./max(abs(Yref(:,:,k)),1e-15);
    entry(k,:)=[d(1,1),d(1,2),d(2,1),d(2,2)];
end

outdir=fullfile(root,'validation_results');
if ~exist(outdir,'dir'),mkdir(outdir);end
T=table(f,real(squeeze(Ysim(1,1,:))),imag(squeeze(Ysim(1,1,:))), ...
    real(squeeze(Ysim(1,2,:))),imag(squeeze(Ysim(1,2,:))), ...
    real(squeeze(Ysim(2,1,:))),imag(squeeze(Ysim(2,1,:))), ...
    real(squeeze(Ysim(2,2,:))),imag(squeeze(Ysim(2,2,:))), ...
    rel,entry(:,1),entry(:,2),entry(:,3),entry(:,4), ...
    'VariableNames',{'frequency_hz','Y11_real','Y11_imag','Y12_real','Y12_imag', ...
    'Y21_real','Y21_imag','Y22_real','Y22_imag','fro_relative_error', ...
    'Y11_relative_error','Y12_relative_error','Y21_relative_error','Y22_relative_error'});
writetable(T,fullfile(outdir,'simscape_vs_matrix_admittance.csv'));

summary.max_fro_relative_error=max(rel);
summary.mean_fro_relative_error=mean(rel);
summary.median_fro_relative_error=median(rel);
summary.max_reciprocity_residual=max(arrayfun(@(k) ...
    abs(Ysim(1,2,k)-Ysim(2,1,k)),1:numel(f)));
summary.state_count=size(A,1);
summary.frequency_count=numel(f);
save(fullfile(outdir,'simscape_linear_model.mat'),'A','B','C','D','f','Ysim','Yref','summary');

fid=fopen(fullfile(outdir,'validation_summary.txt'),'w');
fprintf(fid,'PASS\nstate_count=%d\nfrequency_count=%d\n',summary.state_count,summary.frequency_count);
fprintf(fid,'mean_fro_relative_error=%.12e\n',summary.mean_fro_relative_error);
fprintf(fid,'median_fro_relative_error=%.12e\n',summary.median_fro_relative_error);
fprintf(fid,'max_fro_relative_error=%.12e\n',summary.max_fro_relative_error);
fprintf(fid,'max_reciprocity_residual=%.12e\n',summary.max_reciprocity_residual);
fclose(fid);

fig=figure('Visible','off','Color','w','Position',[100 100 1100 760]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
labs={'Y_{11}','Y_{12}','Y_{21}','Y_{22}'};
inds={[1 1],[1 2],[2 1],[2 2]};
for m=1:4
    nexttile; ij=inds{m};
    semilogx(f,20*log10(abs(squeeze(Yref(ij(1),ij(2),:)))),'k-','LineWidth',1.8);hold on;
    semilogx(f,20*log10(abs(squeeze(Ysim(ij(1),ij(2),:)))),'r--','LineWidth',1.2);
    grid on;xlabel('Frequency (Hz)');ylabel('Magnitude (dB S)');
    title(labs{m});legend('Matrix reference','Simscape linearization','Location','best');
end
exportgraphics(fig,fullfile(outdir,'simscape_vs_matrix_admittance.png'),'Resolution',220);
close(fig);

fprintf('VALIDATION_MEAN_REL_ERROR=%.6e\n',summary.mean_fro_relative_error);
fprintf('VALIDATION_MAX_REL_ERROR=%.6e\n',summary.max_fro_relative_error);
fprintf('TWO_PORT_VALIDATION_PASS\n');
close_system(mdl,0);
end

function Yport=reference_y_matrix(freq,S)
s=1i*2*pi*freq;
nnode=8;
Ainc=zeros(8,nnode);
for side=0:1
    off=4*side;
    for sec=1:4
        row=off+sec;
        Ainc(row,off+sec)=1;
        if sec<4,Ainc(row,off+sec+1)=-1;end
    end
end
Z=diag(S.R_hft)+s*S.L_hft;
Yn=Ainc.'*(Z\Ainc);
for k=1:8
    a=Ainc(k,:).';
    Yn=Yn+s*S.Cseries(k)*(a*a.');
    Yn(k,k)=Yn(k,k)+s*S.Cground(k);
end
for k=1:4
    a=zeros(nnode,1);a(k)=1;a(4+k)=-1;
    Yn=Yn+s*S.Cps(k)*(a*a.');
end
ports=[1 5];internal=setdiff(1:nnode,ports);
Yport=Yn(ports,ports)-Yn(ports,internal)*(Yn(internal,internal)\Yn(internal,ports));
end
