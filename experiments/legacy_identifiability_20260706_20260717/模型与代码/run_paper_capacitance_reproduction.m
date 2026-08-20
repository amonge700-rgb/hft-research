% Reproduce the capacitance-extraction logic of:
% Liu et al., "Experimental Extraction of Parasitic Capacitances for
% High-Frequency Transformers", IEEE TPEL, 2017.
%
% This is a paper-anchored reproduction for prototype 1. It uses the
% paper's three-capacitance admittance matrix, the reported magnetic
% parameters, and the reported characteristic frequencies. It does not use
% digitized raw measurement curves.

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
dataDir = fullfile(projectDir, char([25968 25454]));
noteDir = fullfile(projectDir, char([31508 35760]));
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
if ~exist(noteDir, 'dir'), mkdir(noteDir); end

paper.name = "Liu_2017_TPEL_parasitic_capacitance_prototype1";
paper.n = 4;                 % turn ratio, LV:HV = 1:4
paper.Lm = 94.5e-3;          % H, referred to LV side, Table II
paper.Ls = 96.0e-6;          % H, referred to HV side, Table II
paper.Rs = 1e-3;             % small damping only for plotting/feature search
paper.Rm = 1e12;

% Table III: measured characteristic frequencies for prototype 1.
target.icNames = {'f1_Zoc_peak','f2_Zoc_valley','f3_Zoc_peak'};
target.icFreq = [8.5e3, 1.04e6, 11.0e6];
target.tcNames = {'f1_Zoc_peak','fu_Hv_peak','fzero_Hv_zero'};
target.tcFreq = [8.5e3, 1.04e6, 6.1e6];

% Table IV: reported extracted capacitances.
paper.icCap = [6.37e-12, 215.43e-12, 28.52e-12]; % [Cp Cs Cps]
paper.tcCap = [5.31e-12, 215.59e-12, 28.36e-12];
capNames = {'Cp','Cs','Cps'};

f = logspace(3, 8, 6000);

featIcTableCap = extract_features_from_cap(f, paper.icCap, paper);
featTcTableCap = extract_features_from_cap(f, paper.tcCap, paper);

fitIcCap = fit_cap_from_features(paper.icCap, paper, target.icNames, target.icFreq);
fitTcCap = fit_cap_from_features(paper.tcCap, paper, target.tcNames, target.tcFreq);
formulaIcCap = formula_extract_ic(target.icFreq, paper);
formulaTcCap = formula_extract_tc(target.tcFreq, paper);

featIcFit = extract_features_from_cap(f, fitIcCap, paper);
featTcFit = extract_features_from_cap(f, fitTcCap, paper);
featIcFormula = extract_features_from_cap(f, formulaIcCap, paper);
featTcFormula = extract_features_from_cap(f, formulaTcCap, paper);

rows = [];
rows(1).method = "paper_IC_table";
rows(1).Cp_pF = paper.icCap(1)*1e12;
rows(1).Cs_pF = paper.icCap(2)*1e12;
rows(1).Cps_pF = paper.icCap(3)*1e12;
rows(1).errPct_Cp_vs_paper = 0;
rows(1).errPct_Cs_vs_paper = 0;
rows(1).errPct_Cps_vs_paper = 0;

rows(2).method = "fit_IC_features";
rows(2).Cp_pF = fitIcCap(1)*1e12;
rows(2).Cs_pF = fitIcCap(2)*1e12;
rows(2).Cps_pF = fitIcCap(3)*1e12;
rows(2).errPct_Cp_vs_paper = 100*(fitIcCap(1)-paper.icCap(1))/paper.icCap(1);
rows(2).errPct_Cs_vs_paper = 100*(fitIcCap(2)-paper.icCap(2))/paper.icCap(2);
rows(2).errPct_Cps_vs_paper = 100*(fitIcCap(3)-paper.icCap(3))/paper.icCap(3);

rows(3).method = "paper_TC_table";
rows(3).Cp_pF = paper.tcCap(1)*1e12;
rows(3).Cs_pF = paper.tcCap(2)*1e12;
rows(3).Cps_pF = paper.tcCap(3)*1e12;
rows(3).errPct_Cp_vs_paper = 0;
rows(3).errPct_Cs_vs_paper = 0;
rows(3).errPct_Cps_vs_paper = 0;

rows(4).method = "fit_TC_features";
rows(4).Cp_pF = fitTcCap(1)*1e12;
rows(4).Cs_pF = fitTcCap(2)*1e12;
rows(4).Cps_pF = fitTcCap(3)*1e12;
rows(4).errPct_Cp_vs_paper = 100*(fitTcCap(1)-paper.tcCap(1))/paper.tcCap(1);
rows(4).errPct_Cs_vs_paper = 100*(fitTcCap(2)-paper.tcCap(2))/paper.tcCap(2);
rows(4).errPct_Cps_vs_paper = 100*(fitTcCap(3)-paper.tcCap(3))/paper.tcCap(3);

rows(5).method = "formula_IC_features";
rows(5).Cp_pF = formulaIcCap(1)*1e12;
rows(5).Cs_pF = formulaIcCap(2)*1e12;
rows(5).Cps_pF = formulaIcCap(3)*1e12;
rows(5).errPct_Cp_vs_paper = 100*(formulaIcCap(1)-paper.icCap(1))/paper.icCap(1);
rows(5).errPct_Cs_vs_paper = 100*(formulaIcCap(2)-paper.icCap(2))/paper.icCap(2);
rows(5).errPct_Cps_vs_paper = 100*(formulaIcCap(3)-paper.icCap(3))/paper.icCap(3);

rows(6).method = "formula_TC_features";
rows(6).Cp_pF = formulaTcCap(1)*1e12;
rows(6).Cs_pF = formulaTcCap(2)*1e12;
rows(6).Cps_pF = formulaTcCap(3)*1e12;
rows(6).errPct_Cp_vs_paper = 100*(formulaTcCap(1)-paper.tcCap(1))/paper.tcCap(1);
rows(6).errPct_Cs_vs_paper = 100*(formulaTcCap(2)-paper.tcCap(2))/paper.tcCap(2);
rows(6).errPct_Cps_vs_paper = 100*(formulaTcCap(3)-paper.tcCap(3))/paper.tcCap(3);

summaryTable = struct2table(rows);
writetable(summaryTable, fullfile(dataDir, 'paper_capacitance_reproduction_summary.csv'));

featureRows = build_feature_rows(target, featIcTableCap, featTcTableCap, ...
    featIcFit, featTcFit, featIcFormula, featTcFormula);
featureTable = struct2table(featureRows);
writetable(featureTable, fullfile(dataDir, 'paper_capacitance_feature_check.csv'));

plot_paper_reproduction(f, paper, paper.icCap, paper.tcCap, fitIcCap, fitTcCap, ...
    target, summaryTable, dataDir);
write_reproduction_note(summaryTable, featureTable, paper, dataDir, noteDir);

disp(summaryTable);
disp(featureTable);

function cap = fit_cap_from_features(cap0, paper, featureNames, targetFreq)
    obj = @(x) feature_objective(exp(x), paper, featureNames, targetFreq);
    opts = optimset('Display','off', 'MaxIter', 1000, 'MaxFunEvals', 4000, ...
        'TolX', 1e-11, 'TolFun', 1e-11);
    starts = [
        0, 0, 0
        0.15, -0.08, 0.10
        -0.15, 0.08, -0.10
        0.25, 0.10, -0.20
        -0.25, -0.10, 0.20
    ];
    best = inf;
    cap = cap0;
    for i = 1:size(starts,1)
        [x, val] = fminsearch(obj, log(cap0) + starts(i,:), opts);
        if val < best
            best = val;
            cap = exp(x);
        end
    end
end

function cap = formula_extract_ic(freq, paper)
    f1 = freq(1);
    f2 = freq(2);
    f3 = freq(3);
    A = 1 / ((2*pi*f1)^2 * paper.Lm);
    B = 1 / ((2*pi*f2)^2 * paper.Ls);
    C = paper.n^2 / ((2*pi*f3)^2 * paper.Ls);
    Cps = (C + paper.n^2*B - A) / (2*paper.n);
    Cs = B - Cps;
    Cp = C - Cps;
    cap = [Cp, Cs, Cps];
end

function cap = formula_extract_tc(freq, paper)
    f1 = freq(1);
    fu = freq(2);
    fzero = freq(3);
    A = 1 / ((2*pi*f1)^2 * paper.Lm);
    B = 1 / ((2*pi*fu)^2 * paper.Ls);
    Cps = paper.n / ((2*pi*fzero)^2 * paper.Ls);
    Cs = B - Cps;
    Cp = A - paper.n^2*Cs - (paper.n-1)^2*Cps;
    cap = [Cp, Cs, Cps];
end

function cost = feature_objective(cap, paper, featureNames, targetFreq)
    fSearch = logspace(3, 8, 4000);
    feat = extract_features_from_cap(fSearch, cap, paper);
    pred = zeros(size(targetFreq));
    for k = 1:numel(featureNames)
        pred(k) = feat.(featureNames{k});
    end
    cost = sum((log(pred(:)) - log(targetFreq(:))).^2);
    cost = cost + 1e-4*sum((log(cap(:)) - log([8e-12; 220e-12; 30e-12])).^2);
end

function feat = extract_features_from_cap(f, cap, paper)
    [Zoc, Zsc, Hv] = paper_observables(f, cap, paper);
    zocExt = local_extrema(f, abs(Zoc));
    zscExt = local_extrema(f, abs(Zsc));
    hvExt = local_extrema(f, abs(Hv));

    feat.f1_Zoc_peak = zocExt.peak(1);
    feat.f2_Zoc_valley = zocExt.valley(1);
    if numel(zocExt.peak) >= 2
        feat.f3_Zoc_peak = zocExt.peak(2);
    else
        feat.f3_Zoc_peak = NaN;
    end
    if ~isempty(zscExt.peak)
        feat.f4_Zsc_peak = zscExt.peak(1);
    else
        feat.f4_Zsc_peak = NaN;
    end
    if ~isempty(hvExt.peak)
        feat.fu_Hv_peak = hvExt.peak(1);
    else
        feat.fu_Hv_peak = NaN;
    end
    if ~isempty(hvExt.valley)
        feat.fzero_Hv_zero = hvExt.valley(1);
    else
        feat.fzero_Hv_zero = NaN;
    end
end

function e = local_extrema(f, y)
    peakIdx = [];
    valleyIdx = [];
    yy = log10(y + 1e-300);
    for i = 2:(numel(yy)-1)
        if yy(i) > yy(i-1) && yy(i) > yy(i+1)
            peakIdx(end+1) = i; %#ok<AGROW>
        elseif yy(i) < yy(i-1) && yy(i) < yy(i+1)
            valleyIdx(end+1) = i; %#ok<AGROW>
        end
    end
    e.peak = f(peakIdx);
    e.valley = f(valleyIdx);
end

function [Zoc, Zsc, Hv] = paper_observables(f, cap, paper)
    Cp = cap(1);
    Cs = cap(2);
    Cps = cap(3);
    s = 1i*2*pi*f(:).';
    yl = 1 ./ (paper.Rs + s*paper.Ls);
    ym = 1/paper.Rm + 1 ./ (s*paper.Lm);

    Y11 = paper.n^2 * yl + ym + s*(Cp + Cps);
    Y12 = -paper.n * yl - s*Cps;
    Y22 = yl + s*(Cs + Cps);

    Zsc = 1 ./ Y11;
    Zoc = 1 ./ (Y11 - Y12.^2 ./ Y22);
    Hv = -Y12 ./ Y22;
end

function featureRows = build_feature_rows(target, featIcTableCap, featTcTableCap, featIcFit, featTcFit, featIcFormula, featTcFormula)
    specs = {
        "paper_IC_table", target.icNames, target.icFreq, featIcTableCap
        "fit_IC_features", target.icNames, target.icFreq, featIcFit
        "paper_TC_table", target.tcNames, target.tcFreq, featTcTableCap
        "fit_TC_features", target.tcNames, target.tcFreq, featTcFit
        "formula_IC_features", target.icNames, target.icFreq, featIcFormula
        "formula_TC_features", target.tcNames, target.tcFreq, featTcFormula
    };
    featureRows = [];
    r = 0;
    for s = 1:size(specs,1)
        method = specs{s,1};
        names = specs{s,2};
        targetFreq = specs{s,3};
        feat = specs{s,4};
        for k = 1:numel(names)
            r = r + 1;
            featureRows(r).method = method; %#ok<AGROW>
            featureRows(r).feature = string(names{k});
            featureRows(r).target_Hz = targetFreq(k);
            featureRows(r).predicted_Hz = feat.(names{k});
            featureRows(r).errPct = 100*(featureRows(r).predicted_Hz - targetFreq(k))/targetFreq(k);
        end
    end
end

function plot_paper_reproduction(f, paper, icCap, tcCap, fitIcCap, fitTcCap, target, summaryTable, dataDir)
    [ZocTc, ZscTc, HvTc] = paper_observables(f, tcCap, paper);
    [ZocFit, ZscFit, HvFit] = paper_observables(f, fitTcCap, paper);

    figure('Color','w','Position',[80,80,1050,720]);
    subplot(3,1,1);
    loglog(f, abs(ZocTc), 'k-', 'LineWidth', 1.3); hold on;
    loglog(f, abs(ZocFit), 'r--', 'LineWidth', 1.1);
    xline(target.tcFreq(1), ':', 'f1');
    grid on; ylabel('|Zoc|');
    title('Prototype 1 paper model: IC/TC feature reproduction');
    legend('paper TC capacitance table','fit from TC features','Location','best');

    subplot(3,1,2);
    loglog(f, abs(ZscTc), 'k-', 'LineWidth', 1.3); hold on;
    loglog(f, abs(ZscFit), 'r--', 'LineWidth', 1.1);
    grid on; ylabel('|Zsc|');

    subplot(3,1,3);
    loglog(f, abs(HvTc), 'k-', 'LineWidth', 1.3); hold on;
    loglog(f, abs(HvFit), 'r--', 'LineWidth', 1.1);
    xline(target.tcFreq(2), ':', 'fu');
    xline(target.tcFreq(3), ':', 'fzero');
    grid on; ylabel('|Hv|'); xlabel('Frequency (Hz)');
    saveas(gcf, fullfile(dataDir, 'paper_capacitance_prototype1_curves.png'));

    capMat = [summaryTable.Cp_pF, summaryTable.Cs_pF, summaryTable.Cps_pF];
    figure('Color','w','Position',[80,80,980,470]);
    bar(capMat);
    set(gca, 'XTickLabel', cellstr(summaryTable.method));
    xtickangle(18);
    ylabel('Capacitance (pF)');
    legend({'Cp','Cs','Cps'}, 'Location','best');
    title('Paper table capacitances vs fitted capacitances');
    grid on;
    saveas(gcf, fullfile(dataDir, 'paper_capacitance_fit_comparison.png'));
end

function write_reproduction_note(summaryTable, featureTable, paper, dataDir, noteDir)
    fid = fopen(fullfile(noteDir, char([35770 25991 23492 29983 30005 23481 25552 21462 22797 29616 23454 39564 35760 24405]) + ".md"), 'w');
    fprintf(fid, '# 论文寄生电容提取复现实验记录\n\n');
    fprintf(fid, '对象：Liu et al., Experimental Extraction of Parasitic Capacitances for High-Frequency Transformers, prototype 1.\n\n');
    fprintf(fid, '## 使用的论文数据\n\n');
    fprintf(fid, '- 变比：LV:HV = 1:%g\n', paper.n);
    fprintf(fid, '- Lm = %.4g mH，Ls = %.4g uH\n', paper.Lm*1e3, paper.Ls*1e6);
    fprintf(fid, '- 表 III 特征频率：f1=8.5 kHz, f2/fu=1.04 MHz, f3=11 MHz, fzero=6.1 MHz.\n');
    fprintf(fid, '- 表 IV 寄生电容：IC 法 Cp=6.37 pF, Cs=215.43 pF, Cps=28.52 pF；TC 法 Cp=5.31 pF, Cs=215.59 pF, Cps=28.36 pF.\n\n');
    fprintf(fid, '## 拟合结果\n\n');
    fprintf(fid, '| method | Cp pF | Cs pF | Cps pF | err Cp %% | err Cs %% | err Cps %% |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|\n');
    for i = 1:height(summaryTable)
        fprintf(fid, '| %s | %.4g | %.4g | %.4g | %.4g | %.4g | %.4g |\n', ...
            summaryTable.method(i), summaryTable.Cp_pF(i), summaryTable.Cs_pF(i), summaryTable.Cps_pF(i), ...
            summaryTable.errPct_Cp_vs_paper(i), summaryTable.errPct_Cs_vs_paper(i), summaryTable.errPct_Cps_vs_paper(i));
    end
    fprintf(fid, '\n## 特征频率复核\n\n');
    fprintf(fid, '| method | feature | target Hz | predicted Hz | err %% |\n');
    fprintf(fid, '|---|---|---:|---:|---:|\n');
    for i = 1:height(featureTable)
        fprintf(fid, '| %s | %s | %.5g | %.5g | %.4g |\n', ...
            featureTable.method(i), featureTable.feature(i), featureTable.target_Hz(i), ...
            featureTable.predicted_Hz(i), featureTable.errPct(i));
    end
    fprintf(fid, '\n## 判断\n\n');
    fprintf(fid, '这一步确认了论文三电容导纳矩阵、开路/短路阻抗和电压传递函数之间的对应关系。当前复现使用论文表格特征频率和参数，没有数字化论文图中的原始测量曲线。\n\n');
    fprintf(fid, '后续可以把这套 IC/TC 特征频率提取流程接到在线陡脉冲或 PWM 频响重构上。\n');
    fclose(fid);
end
