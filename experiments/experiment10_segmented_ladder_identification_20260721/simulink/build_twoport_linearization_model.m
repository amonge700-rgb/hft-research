function modelFile = build_twoport_linearization_model()
%BUILD_TWOPORT_LINEARIZATION_MODEL Add two ideal voltage-driven test ports.
%
% Inputs u1,u2 are port voltages in volts. Outputs i1,i2 are currents
% entering the transformer network. The model can therefore be linearized
% once and evaluated as a 2x2 admittance matrix over frequency.

warning('off','all');
root=fileparts(mfilename('fullpath'));
cd(root);
build_segmented_hft_model();

base='hft_segmented_ladder_simscape';
mdl='hft_segmented_ladder_twoport';
load_system(fullfile(root,[base '.slx']));
if bdIsLoaded(mdl),close_system(mdl,0);end
save_system(base,fullfile(root,[mdl '.slx']));
close_system(base,0);
load_system(fullfile(root,[mdl '.slx']));
load_system('hftlib_lib');

libBlocks=find_system('hftlib_lib','SearchDepth',2,'Type','Block');
isTestPort=contains(lower(libBlocks),'voltage') | contains(lower(libBlocks),'test_port');
testPortLib=libBlocks{find(isTestPort,1,'first')};
assert(~isempty(testPortLib),'Could not locate voltage_test_port in hftlib_lib.');

hft=[mdl '/Full 8x8 coupled winding matrix'];
hph=get_param(hft,'PortHandles'); wind=hph.LConn;
ref=[mdl '/Electrical Reference'];
rph=get_param(ref,'PortHandles'); rp=[rph.LConn(:);rph.RConn(:)]; gnd=rp(1);

for port=1:2
    if port==1, node=wind(1); y=160; else, node=wind(9); y=430; end
    in=[mdl sprintf('/v%d',port)];
    conv=[mdl sprintf('/SL_to_PS_v%d',port)];
    src=[mdl sprintf('/Voltage_Test_Port_%d',port)];
    back=[mdl sprintf('/PS_to_SL_i%d',port)];
    out=[mdl sprintf('/i%d',port)];

    add_block('simulink/Ports & Subsystems/In1',in,'Position',[1040 y 1070 y+20]);
    add_block(sprintf('nesl_utility/Simulink-PS\nConverter'),conv, ...
        'Position',[1100 y-10 1170 y+30]);
    add_block(testPortLib,src,'Position',[1220 y-25 1330 y+45]);
    add_block(sprintf('nesl_utility/PS-Simulink\nConverter'),back, ...
        'Position',[1450 y-10 1520 y+30]);
    add_block('simulink/Ports & Subsystems/Out1',out,'Position',[1560 y 1590 y+20]);

    set_param(in,'Port',num2str(port));
    set_param(out,'Port',num2str(port));
    try,set_param(conv,'Unit','V');catch,end
    try,set_param(back,'Unit','A');catch,end

    cph=get_param(conv,'PortHandles');
    sph=get_param(src,'PortHandles');
    bph=get_param(back,'PortHandles');
    inph=get_param(in,'PortHandles');
    outph=get_param(out,'PortHandles');

    add_line(mdl,inph.Outport(1),cph.Inport(1),'autorouting','on');
    % R2021b generated custom block: conserving ports p,n and physical
    % input vcmd appear as LConn(1:3); physical output imeas is RConn(1).
    assert(numel(sph.LConn)==3 && numel(sph.RConn)==1, ...
        'Unexpected voltage_test_port port layout.');
    add_line(mdl,cph.RConn(1),sph.LConn(3),'autorouting','on');
    add_line(mdl,sph.LConn(1),node,'autorouting','on');
    add_line(mdl,sph.LConn(2),gnd,'autorouting','on');
    add_line(mdl,sph.RConn(1),bph.LConn(1),'autorouting','on');
    add_line(mdl,bph.Outport(1),outph.Inport(1),'autorouting','on');
end

set_param(mdl,'Solver','ode23t','StopTime','1e-5');
modelFile=fullfile(root,[mdl '.slx']);
save_system(mdl,modelFile);
fprintf('TWO_PORT_MODEL_SAVED=%s\n',modelFile);
close_system(mdl,0);
end
