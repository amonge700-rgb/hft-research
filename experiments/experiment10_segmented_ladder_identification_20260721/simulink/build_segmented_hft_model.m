function modelFile = build_segmented_hft_model()
%BUILD_SEGMENTED_HFT_MODEL Build the four-section HFT Simscape network.
%
% The topology matches hft_segmented_ladder.py:
%   - 4 primary and 4 secondary series winding sections
%   - full 8x8 reciprocal inductance matrix
%   - longitudinal capacitance in parallel with every section
%   - shunt capacitance from every retained node to reference
%   - four section-wise interwinding capacitances

warning('off','all');
root = fileparts(mfilename('fullpath'));
old = pwd;
cleanup = onCleanup(@() cd(old));
cd(root);

if ~exist(fullfile(root,'hftlib_lib.slx'),'file')
    build_custom_library();
end

load_system('ee_lib');
load_system('hftlib_lib');

mdl = 'hft_segmented_ladder_simscape';
if bdIsLoaded(mdl), close_system(mdl,0); end
if exist(fullfile(root,[mdl '.slx']),'file')
    delete(fullfile(root,[mdl '.slx']));
end
new_system(mdl);
set_param(mdl,'Solver','ode23t','StopTime','1e-5');

% Exact baseline used by the Python experiment.
n = 4;
ratio = 4;
Rp = 0.18/n;
Rs = (0.18/ratio^2)/n;
Llp = 3.6e-6/n;
Lls = (3.6e-6/ratio^2)/n;
Lm = 220e-6;
flux = [ones(4,1); -ones(4,1)/ratio];
R_hft = [repmat(Rp,4,1); repmat(Rs,4,1)];
L_hft = diag([repmat(Llp,4,1); repmat(Lls,4,1)]) ...
      + (Lm/n^2)*(flux*flux.');

Cseries = [repmat(72e-12,4,1); repmat(42e-12,4,1)];
Cground = [7;6;5;4;5;4.5;4;3.5]*1e-12;
Cps = repmat(28.36e-12/n,4,1);

assignin('base','R_hft',R_hft);
assignin('base','L_hft',L_hft);
assignin('base','Lleak_hft',[repmat(Llp,4,1); repmat(Lls,4,1)]);
assignin('base','turns_hft',flux);
assignin('base','Lcommon_hft',Lm/n^2);
assignin('base','Cseries_hft',Cseries);
assignin('base','Cground_hft',Cground);
assignin('base','Cps_hft',Cps);

blocks = find_system('hftlib_lib','SearchDepth',2,'Type','Block');
isCoupled = contains(lower(blocks),'coupled') | contains(lower(blocks),'inductor');
libBlock = blocks{find(isCoupled,1,'first')};
assert(~isempty(libBlock),'Could not locate the coupled-inductor library block.');
hft = [mdl '/Full 8x8 coupled winding matrix'];
add_block(libBlock,hft,'Position',[470 150 700 570]);
try
    set_param(hft,'R','R_hft','Lleak','Lleak_hft', ...
        'a','turns_hft','Lcommon','Lcommon_hft');
catch
    % R2021b generated blocks may store component values in workspace
    % expressions already; keep this explicit diagnostic in the model.
end
hph = get_param(hft,'PortHandles');
wind = hph.LConn;
assert(numel(wind)==16,'Expected 16 conserving ports on custom component.');

% Common electrical reference and solver configuration.
ref = [mdl '/Electrical Reference'];
add_block('ee_lib/Connectors & References/Electrical Reference',ref, ...
          'Position',[850 625 900 675]);
refph = get_param(ref,'PortHandles');
gndPorts = [refph.LConn(:); refph.RConn(:)];
gnd = gndPorts(1);

solver = [mdl '/Solver Configuration'];
add_block('nesl_utility/Solver Configuration',solver, ...
          'Position',[735 625 800 675]);
sph = get_param(solver,'PortHandles');
solverPorts = [sph.LConn(:); sph.RConn(:)];
add_line(mdl,solverPorts(1),gnd,'autorouting','on');

% Winding chains: n_i connects to p_(i+1), final n connects to reference.
for side = 0:1
    offset = side*8;
    for sec = 1:3
        add_line(mdl,wind(offset+2*sec),wind(offset+2*(sec+1)-1), ...
                 'autorouting','on');
    end
    add_line(mdl,wind(offset+8),gnd,'autorouting','on');
end

% Longitudinal capacitances, one in parallel with every winding section.
for k = 1:8
    x = 120 + 80*mod(k-1,4);
    y = 80 + 620*(k>4);
    blk = sprintf('%s/Cs_%d',mdl,k);
    add_block('ee_lib/Passive/Capacitor',blk,'Position',[x y x+45 y+70]);
    set_param(blk,'C',sprintf('Cseries_hft(%d)',k));
    cph = get_param(blk,'PortHandles');
    add_line(mdl,cph.LConn(1),wind(2*k-1),'autorouting','on');
    add_line(mdl,cph.RConn(1),wind(2*k),'autorouting','on');
end

% Node-to-reference capacitances. Retained node k is winding-k positive.
for k = 1:8
    x = 80 + 85*mod(k-1,4);
    y = 260 + 360*(k>4);
    blk = sprintf('%s/Cg_%d',mdl,k);
    add_block('ee_lib/Passive/Capacitor',blk,'Position',[x y x+45 y+70]);
    set_param(blk,'C',sprintf('Cground_hft(%d)',k));
    cph = get_param(blk,'PortHandles');
    add_line(mdl,cph.LConn(1),wind(2*k-1),'autorouting','on');
    add_line(mdl,cph.RConn(1),gnd,'autorouting','on');
end

% Section-wise primary-secondary coupling capacitances.
for sec = 1:4
    y = 185 + 90*(sec-1);
    blk = sprintf('%s/Cps_%d',mdl,sec);
    add_block('ee_lib/Passive/Capacitor',blk,'Position',[930 y 975 y+70]);
    set_param(blk,'C',sprintf('Cps_hft(%d)',sec));
    cph = get_param(blk,'PortHandles');
    pNode = wind(2*sec-1);
    sNode = wind(8+2*sec-1);
    add_line(mdl,cph.LConn(1),pNode,'autorouting','on');
    add_line(mdl,cph.RConn(1),sNode,'autorouting','on');
end

% Human-readable model annotations.
Simulink.Annotation(mdl, ...
    sprintf(['Experiment 10: four-section high-frequency transformer\n' ...
             'Full 8x8 L matrix + Cs + Cg + section-wise Cps\n' ...
             'Primary port: winding 1 positive node; secondary port: winding 5 positive node']));
set_param(mdl,'ZoomFactor','FitSystem');

% Persist all component parameters inside the SLX. This makes the model
% self-contained when it is reopened in a fresh MATLAB desktop session.
mw = get_param(mdl,'ModelWorkspace');
assignin(mw,'R_hft',R_hft);
assignin(mw,'L_hft',L_hft);
assignin(mw,'Lleak_hft',[repmat(Llp,4,1); repmat(Lls,4,1)]);
assignin(mw,'turns_hft',flux);
assignin(mw,'Lcommon_hft',Lm/n^2);
assignin(mw,'Cseries_hft',Cseries);
assignin(mw,'Cground_hft',Cground);
assignin(mw,'Cps_hft',Cps);

modelFile = fullfile(root,[mdl '.slx']);
save_system(mdl,modelFile);

% Save numerical parameters next to the model for independent inspection.
save(fullfile(root,'hft_segmented_ladder_parameters.mat'), ...
     'R_hft','L_hft','Cseries','Cground','Cps','ratio','n');

fprintf('Saved Simscape model: %s\n',modelFile);
fprintf('L symmetry residual: %.3e\n',norm(L_hft-L_hft.','fro'));
fprintf('Minimum eigenvalue of L: %.3e H\n',min(eig(L_hft)));

close_system(mdl,0);
end
