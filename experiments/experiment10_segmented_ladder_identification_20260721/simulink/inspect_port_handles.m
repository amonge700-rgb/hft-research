function inspect_port_handles()
warning('off','all');
load_system('fl_lib'); load_system('simscape');
mdl='inspect_ports_tmp'; if bdIsLoaded(mdl),close_system(mdl,0);end; new_system(mdl);
paths={ ...
 'fl_lib/Electrical/Electrical Sources/Controlled Voltage Source', ...
 'fl_lib/Electrical/Electrical Sensors/Current Sensor', ...
 sprintf('nesl_utility/Simulink-PS\nConverter'), ...
 sprintf('nesl_utility/PS-Simulink\nConverter')};
for k=1:numel(paths)
    b=[mdl '/B' num2str(k)]; add_block(paths{k},b);
    ph=get_param(b,'PortHandles');
    fprintf('BLOCK%d In=%d Out=%d L=%d R=%d\n',k,numel(ph.Inport), ...
        numel(ph.Outport),numel(ph.LConn),numel(ph.RConn));
    fprintf('  In:');fprintf(' %g',ph.Inport);fprintf('\n');
    fprintf('  Out:');fprintf(' %g',ph.Outport);fprintf('\n');
    fprintf('  L:');fprintf(' %g',ph.LConn);fprintf('\n');
    fprintf('  R:');fprintf(' %g',ph.RConn);fprintf('\n');
end
close_system(mdl,0);
end
