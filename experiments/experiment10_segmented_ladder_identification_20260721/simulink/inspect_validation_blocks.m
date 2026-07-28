function inspect_validation_blocks()
warning('off','all');
load_system('fl_lib');
load_system('simscape');
names={'Controlled Voltage Source','Current Sensor','Simulink-PS Converter','PS-Simulink Converter'};
libs={'fl_lib','fl_lib','simscape','simscape'};
for k=1:numel(names)
    p=find_system(libs{k},'LookUnderMasks','all','FollowLinks','on', ...
        'Regexp','on','Name',['^' regexptranslate('escape',names{k}) '$']);
    fprintf('\n%s\n',names{k});
    for j=1:numel(p),fprintf('PATH=%s\n',p{j});end
end
load_system('nesl_utility');
p=find_system('nesl_utility','SearchDepth',2,'Type','Block');
for j=1:numel(p)
    if contains(p{j},'Simulink')
        fprintf('NESL=%s\n',strrep(p{j},newline,'<NL>'));
    end
end
end
