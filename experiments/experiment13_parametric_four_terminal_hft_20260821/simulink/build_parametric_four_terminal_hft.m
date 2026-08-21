function modelFile = build_parametric_four_terminal_hft(p,modelName)
%BUILD_PARAMETRIC_FOUR_TERMINAL_HFT Build an isolated four-terminal model.
% A four-source linear validation fixture drives p+,p-,s+,s- independently.
% The transformer itself does not short either winding terminal to reference.

root=fileparts(mfilename('fullpath'));
old=pwd; cleanup=onCleanup(@() cd(old)); cd(root);
if nargin<2, modelName=sprintf('hft_four_terminal_%dp_%ds',p.Np,p.Ns); end
nb=p.Np+p.Ns;
pkg=fullfile(root,'+hftparam');
generate_full_L_component(nb,pkg);
copyfile(fullfile(root,'..','..','experiment10_segmented_ladder_identification_20260721', ...
    'simulink','+hftlib','voltage_test_port.ssc'),fullfile(pkg,'voltage_test_port.ssc'),'f');
ssc_build('hftparam');
load_system('ee_lib'); load_system('hftparam_lib');

if bdIsLoaded(modelName),close_system(modelName,0);end
outdir=fullfile(root,'generated'); if ~exist(outdir,'dir'),mkdir(outdir);end
modelFile=fullfile(outdir,[modelName '.slx']);
if exist(modelFile,'file'),delete(modelFile);end
new_system(modelName); set_param(modelName,'Solver','ode23t','StopTime','1e-5');

assignin('base','R_exp13',p.R); assignin('base','L_exp13',p.L);
assignin('base','Cs_exp13',p.Cseries); assignin('base','Cgp_exp13',p.Cg_p);
assignin('base','Cgs_exp13',p.Cg_s); assignin('base','Cps_exp13',p.Cps);

libBlocks=find_system('hftparam_lib','SearchDepth',2,'Type','Block');
% ssc_build uses the component description as the visible block name, so
% select the non-test component instead of relying on the source filename.
idx=find(contains(lower(libBlocks),'full') & contains(lower(libBlocks),'reciprocal'),1,'first');
assert(~isempty(idx),'Generated full-L block not found.');
hft=[modelName sprintf('/Full %dx%d L matrix',nb,nb)];
add_block(libBlocks{idx},hft,'Position',[500 160 730 650]);
try,set_param(hft,'R','R_exp13','L','L_exp13');catch,end
hph=get_param(hft,'PortHandles'); wind=hph.LConn;
assert(numel(wind)==2*nb,'Unexpected conserving-port count.');

% Isolated winding chains. Neither final terminal is grounded here.
for k=1:p.Np-1,add_line(modelName,wind(2*k),wind(2*(k+1)-1),'autorouting','on');end
for k=1:p.Ns-1
    a=p.Np+k; b=a+1;
    add_line(modelName,wind(2*a),wind(2*b-1),'autorouting','on');
end

ref=[modelName '/Chassis Reference'];
add_block('ee_lib/Connectors & References/Electrical Reference',ref,'Position',[910 720 960 770]);
rph=get_param(ref,'PortHandles'); rp=[rph.LConn(:);rph.RConn(:)]; gnd=rp(1);
solver=[modelName '/Solver Configuration'];
add_block('nesl_utility/Solver Configuration',solver,'Position',[790 720 860 770]);
sph=get_param(solver,'PortHandles'); sp=[sph.LConn(:);sph.RConn(:)];
add_line(modelName,sp(1),gnd,'autorouting','on');

% Longitudinal capacitors.
for k=1:nb
    blk=sprintf('%s/Cs_%02d',modelName,k);
    add_block('ee_lib/Passive/Capacitor',blk,'Position',[80+55*mod(k-1,max(p.Np,p.Ns)) 60+650*(k>p.Np) 115+55*mod(k-1,max(p.Np,p.Ns)) 115+650*(k>p.Np)]);
    set_param(blk,'C',sprintf('Cs_exp13(%d)',k)); ph=get_param(blk,'PortHandles');
    add_line(modelName,ph.LConn(1),wind(2*k-1),'autorouting','on');
    add_line(modelName,ph.RConn(1),wind(2*k),'autorouting','on');
end

% Node lists include both winding terminals (N+1 nodes per side).
pNodes=[wind(1),arrayfun(@(k)wind(2*k),1:p.Np)];
s0=p.Np; sNodes=[wind(2*(s0+1)-1),arrayfun(@(k)wind(2*(s0+k)),1:p.Ns)];
for k=1:numel(pNodes)
    if p.Cg_p(k)<=0,continue,end
    blk=sprintf('%s/Cgp_%02d',modelName,k); add_block('ee_lib/Passive/Capacitor',blk,'Position',[80+55*(k-1) 270 115+55*(k-1) 325]);
    set_param(blk,'C',sprintf('Cgp_exp13(%d)',k)); ph=get_param(blk,'PortHandles'); add_line(modelName,ph.LConn(1),pNodes(k),'autorouting','on'); add_line(modelName,ph.RConn(1),gnd,'autorouting','on');
end
for k=1:numel(sNodes)
    if p.Cg_s(k)<=0,continue,end
    blk=sprintf('%s/Cgs_%02d',modelName,k); add_block('ee_lib/Passive/Capacitor',blk,'Position',[80+55*(k-1) 545 115+55*(k-1) 600]);
    set_param(blk,'C',sprintf('Cgs_exp13(%d)',k)); ph=get_param(blk,'PortHandles'); add_line(modelName,ph.LConn(1),sNodes(k),'autorouting','on'); add_line(modelName,ph.RConn(1),gnd,'autorouting','on');
end

% General sparse interwinding capacitance matrix.
for i=1:p.Np
    for j=1:p.Ns
        if p.Cps(i,j)<=0,continue,end
        blk=sprintf('%s/Cps_%02d_%02d',modelName,i,j);
        add_block('ee_lib/Passive/Capacitor',blk,'Position',[980+45*mod(j-1,4) 80+55*(i-1)+150*floor((j-1)/4) 1015+45*mod(j-1,4) 125+55*(i-1)+150*floor((j-1)/4)]);
        set_param(blk,'C',sprintf('Cps_exp13(%d,%d)',i,j)); ph=get_param(blk,'PortHandles');
        add_line(modelName,ph.LConn(1),pNodes(i),'autorouting','on'); add_line(modelName,ph.RConn(1),sNodes(j),'autorouting','on');
    end
end

% Four independent terminal test ports. Their current outputs form Y4.
testIdx=find(contains(lower(libBlocks),'ideal voltage-driven'),1,'first');
assert(~isempty(testIdx),'voltage_test_port block not found.');
termNodes=[pNodes(1),pNodes(end),sNodes(1),sNodes(end)];
termNames={'pplus','pminus','splus','sminus'};
for q=1:4
    y=120+145*(q-1); in=[modelName '/v_' termNames{q}]; conv=[modelName '/SLPS_' termNames{q}]; src=[modelName '/Test_' termNames{q}]; back=[modelName '/PSSL_' termNames{q}]; out=[modelName '/i_' termNames{q}];
    add_block('simulink/Ports & Subsystems/In1',in,'Position',[1260 y 1290 y+20]); set_param(in,'Port',num2str(q));
    add_block(sprintf('nesl_utility/Simulink-PS\nConverter'),conv,'Position',[1320 y-10 1390 y+30]);
    add_block(libBlocks{testIdx},src,'Position',[1430 y-25 1535 y+45]);
    add_block(sprintf('nesl_utility/PS-Simulink\nConverter'),back,'Position',[1575 y-10 1645 y+30]);
    add_block('simulink/Ports & Subsystems/Out1',out,'Position',[1680 y 1710 y+20]); set_param(out,'Port',num2str(q));
    iph=get_param(in,'PortHandles'); cph=get_param(conv,'PortHandles'); tph=get_param(src,'PortHandles'); bph=get_param(back,'PortHandles'); oph=get_param(out,'PortHandles');
    add_line(modelName,iph.Outport(1),cph.Inport(1),'autorouting','on'); add_line(modelName,cph.RConn(1),tph.LConn(3),'autorouting','on');
    add_line(modelName,tph.LConn(1),termNodes(q),'autorouting','on'); add_line(modelName,tph.LConn(2),gnd,'autorouting','on');
    add_line(modelName,tph.RConn(1),bph.LConn(1),'autorouting','on'); add_line(modelName,bph.Outport(1),oph.Inport(1),'autorouting','on');
end

try,set_param(modelName,'SimscapeLogType','all','SimscapeLogName','simlog_exp13');catch,end
Simulink.Annotation(modelName,sprintf(['EXP-013 parameterized four-terminal HFT\nNp=%d, Ns=%d, full L, sparse Cps\n' ...
    'Terminal order: p+, p-, s+, s-. Internal branch v/i and capacitor currents are logged.'],p.Np,p.Ns));
mw=get_param(modelName,'ModelWorkspace'); assignin(mw,'R_exp13',p.R);assignin(mw,'L_exp13',p.L);assignin(mw,'Cs_exp13',p.Cseries);assignin(mw,'Cgp_exp13',p.Cg_p);assignin(mw,'Cgs_exp13',p.Cg_s);assignin(mw,'Cps_exp13',p.Cps);
save_system(modelName,modelFile); close_system(modelName,0);
save(fullfile(outdir,[modelName '_parameters.mat']),'p');
end
