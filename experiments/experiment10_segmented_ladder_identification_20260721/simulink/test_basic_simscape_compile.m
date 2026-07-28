function test_basic_simscape_compile()
warning('off','all');
load_system('ee_lib');
mdl='diag_basic_rc';
if bdIsLoaded(mdl),close_system(mdl,0);end
new_system(mdl);
add_block('ee_lib/Passive/Resistor',[mdl '/R']);
add_block('ee_lib/Passive/Capacitor',[mdl '/C']);
add_block('ee_lib/Connectors & References/Electrical Reference',[mdl '/Ref']);
add_block('nesl_utility/Solver Configuration',[mdl '/Solver']);
pr=get_param([mdl '/R'],'PortHandles');
pc=get_param([mdl '/C'],'PortHandles');
pg=get_param([mdl '/Ref'],'PortHandles'); gp=[pg.LConn(:);pg.RConn(:)];
ps=get_param([mdl '/Solver'],'PortHandles'); sp=[ps.LConn(:);ps.RConn(:)];
add_line(mdl,pr.LConn(1),pc.LConn(1));
add_line(mdl,pr.RConn(1),gp(1));
add_line(mdl,pc.RConn(1),gp(1));
add_line(mdl,sp(1),gp(1));
fprintf('BASIC_BUILT\n');
set_param(mdl,'SimulationCommand','update');
fprintf('BASIC_PASS\n');
save_system(mdl,fullfile(fileparts(mfilename('fullpath')),[mdl '.slx']));
close_system(mdl,0);
end
