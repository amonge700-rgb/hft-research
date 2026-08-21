function export_exp13_model_images()
root=fileparts(mfilename('fullpath')); out=fullfile(root,'..','figures');if ~exist(out,'dir'),mkdir(out);end
names={'hft_four_terminal_4p_4s_regression4','hft_four_terminal_8p_8s_candidate8'};
for k=1:numel(names)
    file=fullfile(root,'generated',[names{k} '.slx']);load_system(file);
    try,Simulink.BlockDiagram.arrangeSystem(names{k});save_system(names{k},file);catch,end
    print(['-s' names{k}],'-dpng','-r160',fullfile(out,[names{k} '.png']));
    close_system(names{k},0);
end
end

