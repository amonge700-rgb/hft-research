function modelFile = build_exp14_dab_model()
% Build a separate SPS-DAB model around the EXP-013 8+8 HFT network.
root=fileparts(mfilename('fullpath'));
exp13=fullfile(root,'..','..','experiment13_parametric_four_terminal_hft_20260821','simulink');
addpath(exp13); p=make_exp13_parameters(8,8,'candidate8'); d=make_exp14_settings(); nb=p.Np+p.Ns;
if ~exist(fullfile(root,'+hftparam'),'dir'),copyfile(fullfile(exp13,'+hftparam'),fullfile(root,'+hftparam'),'f');end
if ~exist(fullfile(root,'hftparam_lib.slx'),'file'),ssc_build('hftparam');end
if ~exist(fullfile(root,'hftdab_lib.slx'),'file'),ssc_build('hftdab');end
load_system('ee_lib');load_system('hftparam_lib');load_system('hftdab_lib');
name=d.model_name;if bdIsLoaded(name),close_system(name,0);end
outdir=fullfile(root,'generated');if ~exist(outdir,'dir'),mkdir(outdir);end
modelFile=fullfile(outdir,[name '.slx']);if exist(modelFile,'file'),delete(modelFile);end
new_system(name);set_param(name,'Solver','ode23t','StopTime','d_exp14.stop_cycles*d_exp14.Ts','MaxStep','d_exp14.max_step','RelTol','1e-4');
try,set_param(name,'SimscapeLogType','all','SimscapeLogName','simlog_exp14');catch,end
hlib=find_system('hftparam_lib','SearchDepth',2,'Type','Block');idx=find(contains(lower(hlib),'full')&contains(lower(hlib),'reciprocal'),1);
dlib=find_system('hftdab_lib','SearchDepth',2,'Type','Block');bidx=find(contains(lower(dlib),'lossless full-bridge'),1);sidx=find(contains(lower(dlib),'ideal floating dc'),1);
assert(~isempty(idx)&&~isempty(bidx)&&~isempty(sidx),'Generated blocks missing.');
hft=[name '/Segmented HFT 8p8s'];add_block(hlib{idx},hft,'Position',[630 210 850 660]);try,set_param(hft,'R','R_exp14','L','L_exp14');catch,end
wind=get_param(hft,'PortHandles');wind=wind.LConn;assert(numel(wind)==2*nb);
for k=1:p.Np-1,add_line(name,wind(2*k),wind(2*(k+1)-1),'autorouting','on');end
for k=1:p.Ns-1,a=p.Np+k;add_line(name,wind(2*a),wind(2*(a+1)-1),'autorouting','on');end
pNodes=[wind(1),arrayfun(@(k)wind(2*k),1:p.Np)];sNodes=[wind(2*(p.Np+1)-1),arrayfun(@(k)wind(2*(p.Np+k)),1:p.Ns)];
ref=[name '/Reference'];add_block('ee_lib/Connectors & References/Electrical Reference',ref,'Position',[650 780 700 830]);rph=get_param(ref,'PortHandles');rr=[rph.LConn(:);rph.RConn(:)];gnd=rr(1);
sol=[name '/Solver Configuration'];add_block('nesl_utility/Solver Configuration',sol,'Position',[750 780 825 830]);sph=get_param(sol,'PortHandles');ss=[sph.LConn(:);sph.RConn(:)];add_line(name,ss(1),gnd,'autorouting','on');
set_param(sol,'UseLocalSolver','on','LocalSolverChoice','NE_BACKWARD_EULER_ADVANCER','LocalSolverSampleTime','d_exp14.local_solver_step');
for k=1:nb
 blk=sprintf('%s/Cs_%02d',name,k);add_block('ee_lib/Passive/Capacitor',blk,'Position',[420+24*mod(k-1,8) 80+620*(k>8) 440+24*mod(k-1,8) 115+620*(k>8)]);set_param(blk,'C',sprintf('Cs_exp14(%d)',k));ph=get_param(blk,'PortHandles');add_line(name,ph.LConn(1),wind(2*k-1),'autorouting','on');add_line(name,ph.RConn(1),wind(2*k),'autorouting','on');
end
for k=1:numel(pNodes)
 blk=sprintf('%s/Cgp_%02d',name,k);add_block('ee_lib/Passive/Capacitor',blk,'Position',[400+25*(k-1) 300 420+25*(k-1) 335]);set_param(blk,'C',sprintf('Cgp_exp14(%d)',k));ph=get_param(blk,'PortHandles');add_line(name,ph.LConn(1),pNodes(k),'autorouting','on');add_line(name,ph.RConn(1),gnd,'autorouting','on');
 blk=sprintf('%s/Cgs_%02d',name,k);add_block('ee_lib/Passive/Capacitor',blk,'Position',[400+25*(k-1) 570 420+25*(k-1) 605]);set_param(blk,'C',sprintf('Cgs_exp14(%d)',k));ph=get_param(blk,'PortHandles');add_line(name,ph.LConn(1),sNodes(k),'autorouting','on');add_line(name,ph.RConn(1),gnd,'autorouting','on');
end
for i=1:p.Np,for j=1:p.Ns
 if p.Cps(i,j)<=0,continue,end
 blk=sprintf('%s/Cps_%02d_%02d',name,i,j);add_block('ee_lib/Passive/Capacitor',blk,'Position',[910+25*mod(j-1,4) 100+42*(i-1)+145*floor((j-1)/4) 930+25*mod(j-1,4) 135+42*(i-1)+145*floor((j-1)/4)]);set_param(blk,'C',sprintf('Cps_exp14(%d,%d)',i,j));ph=get_param(blk,'PortHandles');add_line(name,ph.LConn(1),pNodes(i),'autorouting','on');add_line(name,ph.RConn(1),sNodes(j),'autorouting','on');
end,end
bp=[name '/Primary Bridge'];bs=[name '/Secondary Bridge'];vp=[name '/Primary DC'];vs=[name '/Secondary DC'];
add_block(dlib{bidx},bp,'Position',[130 190 270 330]);add_block(dlib{bidx},bs,'Position',[1120 440 1260 580]);add_block(dlib{sidx},vp,'Position',[80 390 190 470]);add_block(dlib{sidx},vs,'Position',[1220 650 1330 730]);
try,set_param(vp,'V','d_exp14.Vdc_p');set_param(vs,'V','d_exp14.Vdc_s');catch,end
pc=get_param(bp,'PortHandles');pc=pc.LConn;sc=get_param(bs,'PortHandles');sc=sc.LConn;pv=get_param(vp,'PortHandles');pv=pv.LConn;sv=get_param(vs,'PortHandles');sv=sv.LConn;
add_line(name,pc(1),pv(1),'autorouting','on');add_line(name,pc(2),pv(2),'autorouting','on');add_line(name,pv(2),gnd,'autorouting','on');add_line(name,pc(3),pNodes(1),'autorouting','on');add_line(name,pc(4),pNodes(end),'autorouting','on');
% EXP-013 encodes the transformer dot convention with negative primary-to-
% secondary mutual terms.  Reverse the secondary AC connection so equal
% referred bridge voltages cancel outside the SPS phase-shift intervals.
add_line(name,sc(3),sNodes(end),'autorouting','on');add_line(name,sc(4),sNodes(1),'autorouting','on');add_line(name,sc(1),sv(1),'autorouting','on');add_line(name,sc(2),sv(2),'autorouting','on');
% The secondary DAB port is electrically floating.  Its absolute common-mode
% potential is therefore undefined at the ideal-model initialisation point.
% A 1-Gohm reference fixes only that numerical gauge; it is not a shared
% power return and its current is negligible at the investigated time scale.
cmref=[name '/Secondary CM reference 1G'];
add_block('ee_lib/Passive/Resistor',cmref,'Position',[1160 750 1240 790]);
set_param(cmref,'R','1e9');cmph=get_param(cmref,'PortHandles');
add_line(name,cmph.LConn(1),sv(2),'autorouting','on');
add_line(name,cmph.RConn(1),gnd,'autorouting','on');
for q=1:2
 if q==1,tag='p';bridge=bp;delay='0';x=40;else,tag='s';bridge=bs;delay='d_exp14.phi_deg/360*d_exp14.Ts';x=1040;end
 pulse=[name '/pulse_' tag];gain=[name '/gain2_' tag];bias=[name '/bias_' tag];rate=[name '/finite_edge_' tag];conv=[name '/SLPS_' tag];
 add_block('simulink/Sources/Pulse Generator',pulse,'Position',[x 80 x+80 110]);set_param(pulse,'Amplitude','1','Period','d_exp14.Ts','PulseWidth','50','PhaseDelay',delay);
 add_block('simulink/Math Operations/Gain',gain,'Position',[x+100 75 x+145 115]);set_param(gain,'Gain','2');add_block('simulink/Math Operations/Bias',bias,'Position',[x+165 75 x+210 115]);set_param(bias,'Bias','-1');
 add_block('simulink/Discontinuities/Rate Limiter',rate,'Position',[x+230 75 x+290 115]);set_param(rate,'RisingSlewLimit','2/d_exp14.transition_time','FallingSlewLimit','-2/d_exp14.transition_time');
 add_block(sprintf('nesl_utility/Simulink-PS\nConverter'),conv,'Position',[x+320 70 x+390 120]);
 ph1=get_param(pulse,'PortHandles');ph2=get_param(gain,'PortHandles');ph3=get_param(bias,'PortHandles');phr=get_param(rate,'PortHandles');ph4=get_param(conv,'PortHandles');bph=get_param(bridge,'PortHandles');add_line(name,ph1.Outport(1),ph2.Inport(1));add_line(name,ph2.Outport(1),ph3.Inport(1));add_line(name,ph3.Outport(1),phr.Inport(1));add_line(name,phr.Outport(1),ph4.Inport(1));add_line(name,ph4.RConn(1),bph.LConn(5),'autorouting','on');
end
bridges={bp,bs};tags={'p','s'};names={'vac','iac','vdc','idc'};
for q=1:2
 ph=get_param(bridges{q},'PortHandles');outs=ph.RConn;
 for z=1:4
  conv=sprintf('%s/PSSL_%s_%s',name,tags{q},names{z});tw=sprintf('%s/to_%s_%s',name,tags{q},names{z});add_block(sprintf('nesl_utility/PS-Simulink\nConverter'),conv,'Position',[1380 100+55*((q-1)*4+z) 1450 130+55*((q-1)*4+z)]);add_block('simulink/Sinks/To Workspace',tw,'Position',[1490 100+55*((q-1)*4+z) 1580 130+55*((q-1)*4+z)]);set_param(tw,'VariableName',sprintf('%s_%s_exp14',tags{q},names{z}),'SaveFormat','Structure With Time');cph=get_param(conv,'PortHandles');tph=get_param(tw,'PortHandles');add_line(name,outs(z),cph.LConn(1),'autorouting','on');add_line(name,cph.Outport(1),tph.Inport(1));
 end
end
mw=get_param(name,'ModelWorkspace');assignin(mw,'R_exp14',p.R);assignin(mw,'L_exp14',p.L);assignin(mw,'Cs_exp14',p.Cseries);assignin(mw,'Cgp_exp14',p.Cg_p);assignin(mw,'Cgs_exp14',p.Cg_s);assignin(mw,'Cps_exp14',p.Cps);assignin(mw,'d_exp14',d);
Simulink.Annotation(name,'EXP-014: separate SPS-DAB switching model; ideal lossless switching bridges; 8+8 segmented four-terminal HFT');save_system(name,modelFile);save(fullfile(outdir,[name '_parameters.mat']),'p','d');close_system(name,0);
end
