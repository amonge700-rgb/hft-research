function inspect_simscape_blocks()
%INSPECT_SIMSCAPE_BLOCKS Print candidate library paths and port layouts.

warning('off','all');
load_system('ee_lib');
load_system('simscape');
load_system('hftlib_lib');

queries = {
    'Capacitor'
    'Electrical Reference'
    'Controlled Voltage Source'
    'Current Sensor'
    'Voltage Sensor'
    'Solver Configuration'
    };

for q = 1:numel(queries)
    fprintf('\n=== %s ===\n', queries{q});
    paths = [ ...
        find_system('ee_lib','LookUnderMasks','all','FollowLinks','on', ...
                    'Regexp','on','Name',['(?i).*' regexptranslate('escape',queries{q}) '.*']); ...
        find_system('simscape','LookUnderMasks','all','FollowLinks','on', ...
                    'Regexp','on','Name',['(?i).*' regexptranslate('escape',queries{q}) '.*']) ...
        ];
    paths = unique(paths);
    for k = 1:min(numel(paths),20)
        fprintf('%s\n', paths{k});
    end
end

mdl = 'inspect_hft_custom_block';
if bdIsLoaded(mdl), close_system(mdl,0); end
new_system(mdl);
blocks = find_system('hftlib_lib','SearchDepth',2,'Type','Block');
libBlock = blocks{1};
fprintf('\nCustom library block: %s\n',libBlock);
add_block(libBlock,[mdl '/HFT']);
ph = get_param([mdl '/HFT'],'PortHandles');
fprintf('\n=== custom block port handles ===\n');
fprintf('LConn: %d, RConn: %d\n',numel(ph.LConn),numel(ph.RConn));
fprintf('LConn handles: '); fprintf('%g ',ph.LConn); fprintf('\n');
fprintf('RConn handles: '); fprintf('%g ',ph.RConn); fprintf('\n');
fprintf('Mask names:\n');
disp(get_param([mdl '/HFT'],'MaskNames'));
close_system(mdl,0);
end
