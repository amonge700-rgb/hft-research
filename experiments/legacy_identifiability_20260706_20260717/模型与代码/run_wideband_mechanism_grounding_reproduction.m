% Wideband mechanism model grounding reproduction.
%
% Paper anchor:
% Liu et al., "Wideband Mechanism Model and Parameter Extracting for
% High-Power High-Voltage High-Frequency Transformers", IEEE TPEL, 2016.
%
% This script uses Table II and Table III to build a four-terminal
% capacitance matrix plus a differential magnetic two-port. It then checks
% whether the resonant-frequency shifts under different grounding
% conditions follow Table IV.
%
% Important limitation:
% The paper used a circuit simulator and the full Fig. 6 model. This script
% is an interpretable table-parameter reproduction, not a full FEM/circuit
% netlist reproduction.

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
dataDir = fullfile(projectDir, char([25968 25454]));
noteDir = fullfile(projectDir, char([31508 35760]));
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
if ~exist(noteDir, 'dir'), mkdir(noteDir); end

paper.name = "Liu_2016_TPEL_wideband_mechanism_model";
paper.n = 91.3;
paper.Lm = 4.587e-3;
paper.Ls0 = 42.45e-3;
paper.Rm = 422.8;
paper.Zs1_R = 3.54e-3;
paper.Zs1_L = 0.158e-3 / (2*pi*1e6);  % Table II reports small imag at high frequency; kept only as weak damping.
paper.Zs2_R = 37.8;
paper.Zs2_L = 21.5e-6 / (2*pi*1e6);

% Table III, equation (9), pF. Balanced one-section estimate.
C.C12 = 8.3e-12;   C.C34 = 13.9e-12;  C.C24 = 93.0e-12;
C.C13 = 95.5e-12;  C.C23 = 61.7e-12;  C.C14 = 59.1e-12;
C.C10 = 161.1e-12; C.C20 = 160.6e-12; C.C30 = 35.3e-12; C.C40 = 29.9e-12;
paper.C = C;

conditions = [
    struct('name',"no_winding_grounded", 'groundedNodes', [],   'targetF1',2.5e3, 'targetF2',74e3)
    struct('name',"lv_winding_grounded", 'groundedNodes', 2,    'targetF1',2.5e3, 'targetF2',74e3)
    struct('name',"hv_winding_grounded", 'groundedNodes', 4,    'targetF1',2.0e3, 'targetF2',60e3)
    struct('name',"both_windings_grounded",'groundedNodes',[2 4],'targetF1',1.8e3, 'targetF2',53e3)
];

f = logspace(2, 6, 2200);
variants = ["leakage_referred_to_HV", "leakage_referred_to_LV"];
allRows = [];
allCurves = struct([]);
row = 0;
curveIdx = 0;

for v = 1:numel(variants)
    variant = variants(v);
    for cidx = 1:numel(conditions)
        cond = conditions(cidx);
        Z = input_impedance_lv(f, paper, cond.groundedNodes, variant);
        feat = extract_open_impedance_features(f, Z);
        row = row + 1;
        allRows(row).variant = variant; %#ok<SAGROW>
        allRows(row).condition = cond.name;
        allRows(row).target_f1_Hz = cond.targetF1;
        allRows(row).target_f2_Hz = cond.targetF2;
        allRows(row).pred_f1_Hz = feat.f1;
        allRows(row).pred_f2_Hz = feat.f2;
        allRows(row).errPct_f1 = 100*(feat.f1-cond.targetF1)/cond.targetF1;
        allRows(row).errPct_f2 = 100*(feat.f2-cond.targetF2)/cond.targetF2;

        curveIdx = curveIdx + 1;
        allCurves(curveIdx).variant = variant; %#ok<SAGROW>
        allCurves(curveIdx).condition = cond.name;
        allCurves(curveIdx).f = f;
        allCurves(curveIdx).Z = Z;
        allCurves(curveIdx).feat = feat;
    end
end

T = struct2table(allRows);
variantScores = groupsummary(T, "variant", "mean", ["errPct_f1","errPct_f2"]);
variantScores.meanAbsErr = (abs(variantScores.mean_errPct_f1) + abs(variantScores.mean_errPct_f2))/2;
[~, bestIdx] = min(variantScores.meanAbsErr);
bestVariant = string(variantScores.variant(bestIdx));

% A light resonant-frequency calibration: one scalar capacitance correction
% chosen from the no-grounded f1. This tests whether the grounding shifts are
% structurally reproduced after fixing the global energy/FEM scaling.
calRows = [];
baseCond = conditions(1);
baseZ = input_impedance_lv(f, paper, baseCond.groundedNodes, bestVariant);
baseFeat = extract_open_impedance_features(f, baseZ);
capScale = (baseFeat.f1 / baseCond.targetF1)^2;
paperCal = paper;
paperCal.C = scale_capacitances(paper.C, capScale);
for cidx = 1:numel(conditions)
    cond = conditions(cidx);
    Z = input_impedance_lv(f, paperCal, cond.groundedNodes, bestVariant);
    feat = extract_open_impedance_features(f, Z);
    calRows(cidx).variant = "calibrated_" + bestVariant; %#ok<SAGROW>
    calRows(cidx).condition = cond.name;
    calRows(cidx).target_f1_Hz = cond.targetF1;
    calRows(cidx).target_f2_Hz = cond.targetF2;
    calRows(cidx).pred_f1_Hz = feat.f1;
    calRows(cidx).pred_f2_Hz = feat.f2;
    calRows(cidx).errPct_f1 = 100*(feat.f1-cond.targetF1)/cond.targetF1;
    calRows(cidx).errPct_f2 = 100*(feat.f2-cond.targetF2)/cond.targetF2;
end
Tcal = struct2table(calRows);

writetable(T, fullfile(dataDir, 'wideband_grounding_reproduction_raw_summary.csv'));
writetable(Tcal, fullfile(dataDir, 'wideband_grounding_reproduction_calibrated_summary.csv'));
save(fullfile(dataDir, 'wideband_grounding_reproduction.mat'), ...
    'paper', 'paperCal', 'conditions', 'f', 'T', 'Tcal', 'variantScores', ...
    'bestVariant', 'capScale', 'allCurves');

plot_wideband_grounding(f, paper, paperCal, conditions, bestVariant, dataDir);
write_wideband_note(T, Tcal, variantScores, bestVariant, capScale, noteDir);

disp(T);
disp(variantScores);
disp(Tcal);

function Z = input_impedance_lv(f, paper, groundedNodes, variant)
    Z = zeros(size(f));
    for k = 1:numel(f)
        Y = total_nodal_admittance(f(k), paper, variant);
        I = zeros(4,1);
        if ismember(2, groundedNodes)
            I(1) = 1;
            measureNodes = [1 2];
        else
            I(1) = 1;
            I(2) = -1;
            measureNodes = [1 2];
        end
        [V, ok] = solve_grounded(Y, I, groundedNodes);
        if ~ok
            Z(k) = NaN;
        else
            Z(k) = V(measureNodes(1)) - V(measureNodes(2));
        end
    end
end

function Y = total_nodal_admittance(f, paper, variant)
    Ycap = capacitance_nodal_admittance(2*pi*f, paper.C);
    Ymag = magnetic_nodal_admittance(2*pi*f, paper, variant);
    Y = Ycap + Ymag;
end

function Ycap = capacitance_nodal_admittance(w, C)
    Cmat = zeros(4,4);
    Cmat = add_pair(Cmat, 1, 2, C.C12);
    Cmat = add_pair(Cmat, 3, 4, C.C34);
    Cmat = add_pair(Cmat, 2, 4, C.C24);
    Cmat = add_pair(Cmat, 1, 3, C.C13);
    Cmat = add_pair(Cmat, 2, 3, C.C23);
    Cmat = add_pair(Cmat, 1, 4, C.C14);
    Cmat(1,1) = Cmat(1,1) + C.C10;
    Cmat(2,2) = Cmat(2,2) + C.C20;
    Cmat(3,3) = Cmat(3,3) + C.C30;
    Cmat(4,4) = Cmat(4,4) + C.C40;
    Ycap = 1i*w*Cmat;
end

function Cmat = add_pair(Cmat, a, b, c)
    Cmat(a,a) = Cmat(a,a) + c;
    Cmat(b,b) = Cmat(b,b) + c;
    Cmat(a,b) = Cmat(a,b) - c;
    Cmat(b,a) = Cmat(b,a) - c;
end

function YmagNode = magnetic_nodal_admittance(w, paper, variant)
    n = paper.n;
    s = 1i*w;
    ym = 1/paper.Rm + 1/(s*paper.Lm);
    switch variant
        case "leakage_referred_to_HV"
            yl = 1/(paper.Zs2_R + s*paper.Ls0);
            Yp = [n^2*yl + ym, -n*yl; -n*yl, yl];
        case "leakage_referred_to_LV"
            yl = 1/(paper.Zs1_R + s*paper.Ls0);
            Yp = [yl + ym, -n*yl; -n*yl, n^2*yl];
        otherwise
            error('Unknown variant');
    end
    B = [1 -1 0 0; 0 0 1 -1];
    YmagNode = B' * Yp * B;
end

function [V, ok] = solve_grounded(Y, I, groundedNodes)
    keep = setdiff(1:4, groundedNodes);
    V = zeros(4,1);
    Yk = Y(keep, keep);
    Ik = I(keep);
    ok = rcond(Yk) > 1e-14;
    if ok
        V(keep) = Yk \ Ik;
    end
end

function feat = extract_open_impedance_features(f, Z)
    y = smoothdata(log10(abs(Z)+1e-300), 'movmean', 9);
    idx1 = find(f >= 0.8e3 & f <= 8e3 & isfinite(y));
    idx2 = find(f >= 20e3 & f <= 150e3 & isfinite(y));
    [~, i1] = max(y(idx1));
    [~, i2] = min(y(idx2));
    feat.f1 = f(idx1(i1));
    feat.f2 = f(idx2(i2));
end

function C2 = scale_capacitances(C, scale)
    names = fieldnames(C);
    C2 = C;
    for i = 1:numel(names)
        C2.(names{i}) = C.(names{i}) * scale;
    end
end

function plot_wideband_grounding(f, paper, paperCal, conditions, bestVariant, dataDir)
    figure('Color','w','Position',[80,80,1050,760]);
    for cidx = 1:numel(conditions)
        cond = conditions(cidx);
        Zraw = input_impedance_lv(f, paper, cond.groundedNodes, bestVariant);
        Zcal = input_impedance_lv(f, paperCal, cond.groundedNodes, bestVariant);
        subplot(2,2,cidx);
        semilogx(f, 20*log10(abs(Zraw)), 'Color', [0.4 0.4 0.4], 'LineWidth', 1.0); hold on;
        semilogx(f, 20*log10(abs(Zcal)), 'b--', 'LineWidth', 1.2);
        xline(cond.targetF1, ':r', 'f1 target');
        xline(cond.targetF2, ':m', 'f2 target');
        grid on;
        title(strrep(cond.name, '_', '\_'));
        xlabel('Frequency (Hz)');
        ylabel('|Z_{1oc}| (dB-ohm)');
        if cidx == 1
            legend('raw Table II/III model','capacitance-scaled model','Location','best');
        end
    end
    saveas(gcf, fullfile(dataDir, 'wideband_grounding_reproduction_curves.png'));

    figure('Color','w','Position',[80,80,900,420]);
    names = string({conditions.name});
    targetF1 = [conditions.targetF1] / 1e3;
    targetF2 = [conditions.targetF2] / 1e3;
    predF1 = zeros(1,numel(conditions));
    predF2 = zeros(1,numel(conditions));
    for cidx = 1:numel(conditions)
        Zcal = input_impedance_lv(f, paperCal, conditions(cidx).groundedNodes, bestVariant);
        feat = extract_open_impedance_features(f, Zcal);
        predF1(cidx) = feat.f1 / 1e3;
        predF2(cidx) = feat.f2 / 1e3;
    end
    tiledlayout(1,2);
    nexttile;
    bar([targetF1(:), predF1(:)]);
    set(gca,'XTickLabel',cellstr(names)); xtickangle(25);
    ylabel('f1 (kHz)'); grid on; legend('paper Table IV','reproduction');
    nexttile;
    bar([targetF2(:), predF2(:)]);
    set(gca,'XTickLabel',cellstr(names)); xtickangle(25);
    ylabel('f2 (kHz)'); grid on; legend('paper Table IV','reproduction');
    saveas(gcf, fullfile(dataDir, 'wideband_grounding_frequency_comparison.png'));
end

function write_wideband_note(T, Tcal, variantScores, bestVariant, capScale, noteDir)
    path = fullfile(noteDir, char([23485 39057 26426 29702 27169 22411 25509 22320 26465 20214 22797 29616 35760 24405]) + ".md");
    fid = fopen(path, 'w');
    fprintf(fid, '# 宽频机理模型接地条件复现记录\n\n');
    fprintf(fid, '对象：Liu et al., Wideband Mechanism Model and Parameter Extracting for High-Power High-Voltage High-Frequency Transformers, IEEE TPEL 2016.\n\n');
    fprintf(fid, '## 使用数据\n\n');
    fprintf(fid, '- Table I: 20 kHz, 30 kVA, ratio 1:91.3, LV 12 turns, HV 1096 turns.\n');
    fprintf(fid, '- Table II: Lm=4.587 mH, Ls0=42.45 mH, Rm=422.8 Ohm, Zs1/Zs2.\n');
    fprintf(fid, '- Table III: C12, C34, C24, C13, C23, C14, C10, C20, C30, C40.\n');
    fprintf(fid, '- Table IV: grounding-condition resonant frequencies.\n\n');
    fprintf(fid, '## 模型说明\n\n');
    fprintf(fid, '本脚本把 Table III 构造成四端子电容矩阵，并把磁性部分近似成差模两端口网络。它不是作者完整 Fig. 6 电路仿真器，因此结果用于趋势复现和下一步在线 PWM 目标构造，不作为严格原文复现实验。\n\n');
    fprintf(fid, 'Best orientation: %s. Capacitance global scale after f1 calibration: %.4g.\n\n', bestVariant, capScale);
    fprintf(fid, '## Raw table model\n\n');
    write_table(fid, T);
    fprintf(fid, '\n## Calibrated model\n\n');
    write_table(fid, Tcal);
    fprintf(fid, '\n## 判断\n\n');
    fprintf(fid, '该文献数据适合补充结构/FEM参数来源。后续可用该模型生成更接近结构机理的对象，再接入在线 PWM 特征频率恢复实验。主线仍然是在线 PWM；该实验负责让在线 PWM 的对象更像文献中的高频变压器，而不是自定义曲线。\n');
    fclose(fid);
end

function write_table(fid, T)
    fprintf(fid, '| variant | condition | target f1 | pred f1 | err f1 %% | target f2 | pred f2 | err f2 %% |\n');
    fprintf(fid, '|---|---|---:|---:|---:|---:|---:|---:|\n');
    for i = 1:height(T)
        fprintf(fid, '| %s | %s | %.5g | %.5g | %.4g | %.5g | %.5g | %.4g |\n', ...
            T.variant(i), T.condition(i), T.target_f1_Hz(i), T.pred_f1_Hz(i), T.errPct_f1(i), ...
            T.target_f2_Hz(i), T.pred_f2_Hz(i), T.errPct_f2(i));
    end
end

