function diagnose_compile_stages()
%DIAGNOSE_COMPILE_STAGES Compile progressively richer physical networks.
% The status file is flushed after every stage so a timeout remains useful.

warning('off','all');
root = fileparts(mfilename('fullpath'));
cd(root);
load_system('ee_lib');
load_system('hftlib_lib');

logPath = fullfile(root,'compile_stage_diagnostics.txt');
fid = fopen(logPath,'w');
cleanup = onCleanup(@() fclose(fid));

stages = {'core','core_cs','core_cs_cg','full'};
for s = 1:numel(stages)
    stage = stages{s};
    fprintf(fid,'START %s %s\n',stage,datestr(now,30));
    fprintf('START_STAGE_%s\n',upper(stage));
    mdl = ['diag_' stage];
    try
        make_stage(mdl,stage);
        fprintf(fid,'BUILT %s %s\n',stage,datestr(now,30));
        set_param(mdl,'SimulationCommand','update');
        fprintf(fid,'PASS %s %s\n',stage,datestr(now,30));
        fprintf('PASS_STAGE_%s\n',upper(stage));
        close_system(mdl,0);
    catch ME
        fprintf(fid,'FAIL %s %s\n%s\n',stage,datestr(now,30), ...
            getReport(ME,'extended','hyperlinks','off'));
        fprintf('FAIL_STAGE_%s: %s\n',upper(stage),ME.message);
        if bdIsLoaded(mdl), close_system(mdl,0); end
        rethrow(ME);
    end
end
end

function make_stage(mdl,stage)
new_system(mdl);
set_param(mdl,'Solver','ode23t','StopTime','1e-5');

n = 4; ratio = 4;
Rp = 0.18/n; Rs = (0.18/ratio^2)/n;
Llp = 3.6e-6/n; Lls = (3.6e-6/ratio^2)/n; Lm = 220e-6;
flux = [ones(4,1); -ones(4,1)/ratio];
R_hft = [repmat(Rp,4,1); repmat(Rs,4,1)];
L_hft = diag([repmat(Llp,4,1); repmat(Lls,4,1)]) ...
      + (Lm/n^2)*(flux*flux.');
assignin('base','R_hft',R_hft);
assignin('base','L_hft',L_hft);
assignin('base','Lleak_hft',[repmat(Llp,4,1); repmat(Lls,4,1)]);
assignin('base','turns_hft',flux);
assignin('base','Lcommon_hft',Lm/n^2);

blocks = find_system('hftlib_lib','SearchDepth',2,'Type','Block');
hft = [mdl '/HFT'];
add_block(blocks{1},hft,'Position',[400 100 650 500]);
set_param(hft,'R','R_hft','Lleak','Lleak_hft', ...
    'a','turns_hft','Lcommon','Lcommon_hft');
ph = get_param(hft,'PortHandles'); wind = ph.LConn;

ref = [mdl '/Reference'];
add_block('ee_lib/Connectors & References/Electrical Reference',ref, ...
    'Position',[760 520 810 570]);
rph = get_param(ref,'PortHandles');
rp = [rph.LConn(:);rph.RConn(:)]; gnd = rp(1);
solver = [mdl '/Solver'];
add_block('nesl_utility/Solver Configuration',solver, ...
    'Position',[670 520 730 570]);
sph = get_param(solver,'PortHandles');
sp = [sph.LConn(:);sph.RConn(:)];
add_line(mdl,sp(1),gnd,'autorouting','on');

for side=0:1
    off=8*side;
    for sec=1:3
        add_line(mdl,wind(off+2*sec),wind(off+2*(sec+1)-1),'autorouting','on');
    end
    add_line(mdl,wind(off+8),gnd,'autorouting','on');
end

if any(strcmp(stage,{'core_cs','core_cs_cg','full'}))
    Cs=[repmat(72e-12,4,1);repmat(42e-12,4,1)];
    for k=1:8
        b=sprintf('%s/Cs%d',mdl,k);
        add_block('ee_lib/Passive/Capacitor',b,'Position',[80 50+65*k 120 90+65*k]);
        set_param(b,'C',num2str(Cs(k),17));
        cp=get_param(b,'PortHandles');
        add_line(mdl,cp.LConn(1),wind(2*k-1),'autorouting','on');
        add_line(mdl,cp.RConn(1),wind(2*k),'autorouting','on');
    end
end

if any(strcmp(stage,{'core_cs_cg','full'}))
    Cg=[7;6;5;4;5;4.5;4;3.5]*1e-12;
    for k=1:8
        b=sprintf('%s/Cg%d',mdl,k);
        add_block('ee_lib/Passive/Capacitor',b,'Position',[850 50+65*k 890 90+65*k]);
        set_param(b,'C',num2str(Cg(k),17));
        cp=get_param(b,'PortHandles');
        add_line(mdl,cp.LConn(1),wind(2*k-1),'autorouting','on');
        add_line(mdl,cp.RConn(1),gnd,'autorouting','on');
    end
end

if strcmp(stage,'full')
    Cps=28.36e-12/n;
    for k=1:4
        b=sprintf('%s/Cps%d',mdl,k);
        add_block('ee_lib/Passive/Capacitor',b,'Position',[940 100+90*k 980 150+90*k]);
        set_param(b,'C',num2str(Cps,17));
        cp=get_param(b,'PortHandles');
        add_line(mdl,cp.LConn(1),wind(2*k-1),'autorouting','on');
        add_line(mdl,cp.RConn(1),wind(8+2*k-1),'autorouting','on');
    end
end
save_system(mdl,fullfile(fileparts(mfilename('fullpath')),[mdl '.slx']));
end
