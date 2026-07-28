function test_basic_simulink_compile()
warning('off','all');
mdl='diag_basic_simulink';
if bdIsLoaded(mdl),close_system(mdl,0);end
new_system(mdl);
add_block('simulink/Sources/Sine Wave',[mdl '/Input']);
add_block('simulink/Math Operations/Gain',[mdl '/Gain']);
add_block('simulink/Sinks/Out1',[mdl '/Output']);
add_line(mdl,'Input/1','Gain/1');
add_line(mdl,'Gain/1','Output/1');
set_param(mdl,'SimulationCommand','update');
fprintf('BASIC_SIMULINK_PASS\n');
save_system(mdl,fullfile(fileparts(mfilename('fullpath')),[mdl '.slx']));
close_system(mdl,0);
end
