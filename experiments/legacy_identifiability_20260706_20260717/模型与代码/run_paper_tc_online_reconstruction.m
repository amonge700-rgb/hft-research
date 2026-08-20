% Online reconstruction trial for Liu 2017 prototype 1 TC method.
%
% Goal:
%   Paper TC method:
%       offline Zoc/Hv features -> f1, fu, fzero -> Cp, Cs, Cps
%   This trial:
%       time-domain v1/i1/v2 under pulse/PRBS/PWM excitation
%       -> FFT transfer reconstruction
%       -> f1, fu, fzero
%       -> Cp, Cs, Cps
%
% This script uses the paper's prototype-1 parameters and capacitance table.
% It does not digitize raw measured curves from the paper.

clear; clc; close all;
rng(20260712);

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
dataDir = fullfile(projectDir, char([25968 25454]));
noteDir = fullfile(projectDir, char([31508 35760]));
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
if ~exist(noteDir, 'dir'), mkdir(noteDir); end

paper.n = 4;
paper.Lm = 94.5e-3;
paper.Ls = 96.0e-6;
paper.Rs = 0.08;      % damping for finite simulated peaks
paper.Rm = 1e9;
paper.tcCap = [5.31e-12, 215.59e-12, 28.36e-12]; % Cp, Cs, Cps
paper.targetFreq = [8.5e3, 1.04e6, 6.1e6];       % f1, fu, fzero
featureNames = {'f1','fu','fzero'};
capNames = {'Cp','Cs','Cps'};

sim.fs = 100e6;
sim.N = 2^19;
sim.nAvg = 16;
sim.inputNoiseRel = 0.001;
sim.outputNoiseRel = 0.004;
sim.fGrid = logspace(3, log10(20e6), 5000);

[ZocTrue, HvTrue, YinTrue] = paper_observables(sim.fGrid, paper.tcCap, paper);
idealFeat = extract_tc_features(sim.fGrid, ZocTrue, HvTrue);
idealCap = formula_extract_tc([idealFeat.f1, idealFeat.fu, idealFeat.fzero], paper);

methods = ["paper_TC_reference", "single_steep_pulse", "random_prbs_edges", ...
    "targeted_binary_pwm", "swept_binary_pwm"];
rows = [];
featureRows = [];
curves = struct([]);

for m = 1:numel(methods)
    method = methods(m);
    if method == "paper_TC_reference"
        fUse = sim.fGrid;
        ZocEst = ZocTrue;
        HvEst = HvTrue;
        inputEnergy = ones(size(fUse));
    else
        rec = reconstruct_from_time_waveform(method, paper, sim);
        fUse = rec.fGrid;
        ZocEst = rec.Zoc;
        HvEst = rec.Hv;
        inputEnergy = rec.inputEnergy;
    end

    if method == "paper_TC_reference"
        feat.f1 = paper.targetFreq(1);
        feat.fu = paper.targetFreq(2);
        feat.fzero = paper.targetFreq(3);
        capHat = paper.tcCap;
    else
        feat = extract_tc_features(fUse, ZocEst, HvEst);
        capHat = formula_extract_tc([feat.f1, feat.fu, feat.fzero], paper);
    end
    capErr = 100 * (capHat - paper.tcCap) ./ paper.tcCap;
    freqHat = [feat.f1, feat.fu, feat.fzero];
    freqErr = 100 * (freqHat - paper.targetFreq) ./ paper.targetFreq;

    rows(m).method = method; %#ok<SAGROW>
    rows(m).f1_Hz = feat.f1;
    rows(m).fu_Hz = feat.fu;
    rows(m).fzero_Hz = feat.fzero;
    rows(m).errPct_f1 = freqErr(1);
    rows(m).errPct_fu = freqErr(2);
    rows(m).errPct_fzero = freqErr(3);
    rows(m).Cp_pF = capHat(1)*1e12;
    rows(m).Cs_pF = capHat(2)*1e12;
    rows(m).Cps_pF = capHat(3)*1e12;
    rows(m).errPct_Cp = capErr(1);
    rows(m).errPct_Cs = capErr(2);
    rows(m).errPct_Cps = capErr(3);

    for k = 1:numel(featureNames)
        idx = (m-1)*numel(featureNames) + k;
        featureRows(idx).method = method; %#ok<SAGROW>
        featureRows(idx).feature = string(featureNames{k});
        featureRows(idx).target_Hz = paper.targetFreq(k);
        featureRows(idx).estimated_Hz = freqHat(k);
        featureRows(idx).errPct = freqErr(k);
    end

    curves(m).method = method; %#ok<SAGROW>
    curves(m).f = fUse;
    curves(m).Zoc = ZocEst;
    curves(m).Hv = HvEst;
    curves(m).inputEnergy = inputEnergy;
end

summaryTable = struct2table(rows);
featureTable = struct2table(featureRows);
writetable(summaryTable, fullfile(dataDir, 'paper_tc_online_reconstruction_summary.csv'));
writetable(featureTable, fullfile(dataDir, 'paper_tc_online_feature_check.csv'));
save(fullfile(dataDir, 'paper_tc_online_reconstruction.mat'), ...
    'summaryTable', 'featureTable', 'paper', 'sim', 'curves', ...
    'ZocTrue', 'HvTrue', 'YinTrue', 'idealFeat', 'idealCap');

plot_online_results(sim.fGrid, ZocTrue, HvTrue, paper, curves, summaryTable, capNames, dataDir);
write_online_note(summaryTable, featureTable, paper, sim, noteDir);

disp(summaryTable);
disp(featureTable);

function rec = reconstruct_from_time_waveform(method, paper, sim)
    fs = sim.fs;
    N = sim.N;
    nPos = floor(N/2) + 1;
    fPos = (0:nPos-1) * fs/N;
    [~, HvPos, YinPos] = paper_observables(fPos, paper.tcCap, paper);

    Svv = zeros(1, nPos);
    Siv = zeros(1, nPos);
    Sv2v = zeros(1, nPos);

    for a = 1:sim.nAvg
        v = make_excitation(method, sim, a);
        V = fft(v);
        Ifull = build_full_spectrum(YinPos, N) .* V;
        V2full = build_full_spectrum(HvPos, N) .* V;
        i1 = real(ifft(Ifull));
        v2 = real(ifft(V2full));

        vMeas = v + sim.inputNoiseRel * max(std(v), 1e-12) * randn(size(v));
        iMeas = i1 + sim.outputNoiseRel * max(std(i1), 1e-12) * randn(size(i1));
        v2Meas = v2 + sim.outputNoiseRel * max(std(v2), 1e-12) * randn(size(v2));

        Vm = fft(vMeas);
        Im = fft(iMeas);
        V2m = fft(v2Meas);
        Vp = Vm(1:nPos);
        Ip = Im(1:nPos);
        V2p = V2m(1:nPos);

        Svv = Svv + abs(Vp).^2;
        Siv = Siv + Ip .* conj(Vp);
        Sv2v = Sv2v + V2p .* conj(Vp);
    end

    YinEst = Siv ./ max(Svv, 1e-24);
    HvEst = Sv2v ./ max(Svv, 1e-24);
    ZocEst = 1 ./ max_complex(YinEst, 1e-18);
    energy = Svv / max(Svv);

    valid = fPos >= min(sim.fGrid) & fPos <= max(sim.fGrid) & energy > 1e-8;
    rec.fGrid = sim.fGrid;
    rec.Zoc = interp_complex(fPos(valid), ZocEst(valid), sim.fGrid);
    rec.Hv = interp_complex(fPos(valid), HvEst(valid), sim.fGrid);
    rec.inputEnergy = interp1(log10(fPos(valid)), energy(valid), log10(sim.fGrid), 'linear', 0);
end

function v = make_excitation(method, sim, avgIdx)
    fs = sim.fs;
    N = sim.N;
    t = (0:N-1) / fs;
    switch method
        case "single_steep_pulse"
            t0 = 0.35e-3 + 2e-6*(avgIdx-1);
            width = 1.0e-3;
            tr = 50e-9;
            v = 0.5*(tanh((t-t0)/tr) - tanh((t-t0-width)/tr));
        case "random_prbs_edges"
            chip = 8;
            levels = 2*(rand(1, ceil(N/chip)) > 0.5) - 1;
            v = repelem(levels, chip);
            v = v(1:N);
        case "targeted_binary_pwm"
            f1 = 8.5e3;
            fu = 1.04e6;
            fz = 6.1e6;
            phase = 2*pi*rand(1,3);
            x = 0.65*sin(2*pi*f1*t + phase(1)) + ...
                0.70*sin(2*pi*fu*t + phase(2)) + ...
                0.85*sin(2*pi*fz*t + phase(3));
            x = x + 0.20*randn(size(x));
            v = ones(size(x));
            v(x < 0) = -1;
        case "swept_binary_pwm"
            fStart = 0.15e6;
            fStop = 9.0e6;
            T = N / fs;
            k = (fStop - fStart) / T;
            phaseSweep = 2*pi*(fStart*t + 0.5*k*t.^2);
            lowPhase = 2*pi*rand;
            x = sin(phaseSweep) + 0.35*sin(2*pi*8.5e3*t + lowPhase);
            x = x + 0.05*randn(size(x));
            v = ones(size(x));
            v(x < 0) = -1;
        otherwise
            error('Unknown method: %s', method);
    end
    v = v - mean(v);
    v = v ./ max(abs(v) + 1e-12);
end

function [Zoc, Hv, YinOc] = paper_observables(f, cap, paper)
    f = f(:).';
    f(1) = max(f(1), 1e-3);
    Cp = cap(1);
    Cs = cap(2);
    Cps = cap(3);
    s = 1i*2*pi*f;
    yl = 1 ./ (paper.Rs + s*paper.Ls);
    ym = 1/paper.Rm + 1 ./ (s*paper.Lm);
    Y11 = paper.n^2 * yl + ym + s*(Cp + Cps);
    Y12 = -paper.n * yl - s*Cps;
    Y22 = yl + s*(Cs + Cps);
    YinOc = Y11 - Y12.^2 ./ Y22;
    Zoc = 1 ./ YinOc;
    Hv = -Y12 ./ Y22;
end

function feat = extract_tc_features(f, Zoc, Hv)
    f = f(:).';
    z = smooth_log_abs(Zoc);
    h = smooth_log_abs(Hv);
    feat.f1 = peak_in_band(f, z, 4e3, 25e3);
    feat.fu = peak_in_band(f, h, 0.45e6, 2.2e6);
    feat.fzero = valley_in_band(f, h, 3.0e6, 10.0e6);
end

function y = smooth_log_abs(x)
    y = log10(abs(x(:).') + 1e-300);
    y(~isfinite(y)) = nanmedian(y(isfinite(y)));
    y = movmean(y, 5, 'omitnan');
end

function fp = peak_in_band(f, y, fmin, fmax)
    idx = find(f >= fmin & f <= fmax & isfinite(y));
    [~, k] = max(y(idx));
    fp = f(idx(k));
end

function fv = valley_in_band(f, y, fmin, fmax)
    idx = find(f >= fmin & f <= fmax & isfinite(y));
    [~, k] = min(y(idx));
    fv = f(idx(k));
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

function Hfull = build_full_spectrum(Hpos, N)
    nPos = floor(N/2)+1;
    Hpos = Hpos(:).';
    Hpos(1) = Hpos(2);
    if mod(N,2) == 0
        Hneg = conj(Hpos((nPos-1):-1:2));
    else
        Hneg = conj(Hpos(nPos:-1:2));
    end
    Hfull = [Hpos, Hneg];
end

function z = max_complex(z, floorVal)
    small = abs(z) < floorVal;
    z(small) = floorVal;
end

function yi = interp_complex(f, y, fi)
    good = f > 0 & isfinite(real(y)) & isfinite(imag(y));
    yi = interp1(log10(f(good)), y(good), log10(fi), 'linear', nan);
end

function plot_online_results(f, ZocTrue, HvTrue, paper, curves, T, capNames, dataDir)
    colors = lines(numel(curves));
    figure('Color','w','Position',[80,80,1100,720]);
    subplot(2,1,1);
    loglog(f, abs(ZocTrue), 'k-', 'LineWidth', 1.5); hold on;
    for m = 2:numel(curves)
        loglog(curves(m).f, abs(curves(m).Zoc), '--', 'LineWidth', 1.0, 'Color', colors(m,:));
    end
    xline(paper.targetFreq(1), ':', 'f1');
    grid on; ylabel('|Zoc|');
    title('Online reconstruction of paper TC observables');
    legend(['ideal', cellstr([curves(2:end).method])], 'Location','best');

    subplot(2,1,2);
    loglog(f, abs(HvTrue), 'k-', 'LineWidth', 1.5); hold on;
    for m = 2:numel(curves)
        loglog(curves(m).f, abs(curves(m).Hv), '--', 'LineWidth', 1.0, 'Color', colors(m,:));
    end
    xline(paper.targetFreq(2), ':', 'fu');
    xline(paper.targetFreq(3), ':', 'fzero');
    grid on; ylabel('|Hv|'); xlabel('Frequency (Hz)');
    saveas(gcf, fullfile(dataDir, 'paper_tc_online_reconstructed_curves.png'));

    figure('Color','w','Position',[80,80,1100,430]);
    for m = 2:numel(curves)
        semilogx(curves(m).f, curves(m).inputEnergy, 'LineWidth', 1.1); hold on;
    end
    xline(paper.targetFreq(1), ':', 'f1');
    xline(paper.targetFreq(2), ':', 'fu');
    xline(paper.targetFreq(3), ':', 'fzero');
    ylim([0, 1.05]); grid on;
    xlabel('Frequency (Hz)');
    ylabel('Normalized input spectral energy');
    title('Excitation spectrum available for online TC reconstruction');
    legend(cellstr([curves(2:end).method]), 'Location','best');
    saveas(gcf, fullfile(dataDir, 'paper_tc_online_excitation_spectra.png'));

    errMat = [T.errPct_Cp, T.errPct_Cs, T.errPct_Cps];
    figure('Color','w','Position',[80,80,1050,460]);
    bar(errMat);
    set(gca, 'XTickLabel', cellstr(T.method));
    xtickangle(18);
    ylabel('Capacitance error (%)');
    legend(capNames, 'Location','best');
    title('TC capacitance extraction error from online reconstructed features');
    grid on;
    saveas(gcf, fullfile(dataDir, 'paper_tc_online_capacitance_error.png'));
end

function write_online_note(T, F, paper, sim, noteDir)
    path = fullfile(noteDir, char([35770 25991 84 67 26041 27861 22312 32447 21270 38497 33033 20914 20914 23454 39564 35760 24405]) + ".md");
    fid = fopen(path, 'w');
    fprintf(fid, '# 论文 TC 方法在线化脉冲/PWM 实验记录\n\n');
    fprintf(fid, '对象：Liu 2017 prototype 1，使用论文表 II/III/IV 的 Lm、Ls、f1、fu、fzero 和 TC 电容值。\n\n');
    fprintf(fid, '## 实验设置\n\n');
    fprintf(fid, '- fs = %.3g Hz, N = %d, averages = %d\n', sim.fs, sim.N, sim.nAvg);
    fprintf(fid, '- 目标频率：f1=%.4g Hz, fu=%.4g Hz, fzero=%.4g Hz\n', paper.targetFreq);
    fprintf(fid, '- 论文 TC 电容：Cp=5.31 pF, Cs=215.59 pF, Cps=28.36 pF\n\n');
    fprintf(fid, '## 参数反演结果\n\n');
    fprintf(fid, '| method | f1 Hz | fu Hz | fzero Hz | Cp pF | Cs pF | Cps pF | err Cp %% | err Cs %% | err Cps %% |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n');
    for i = 1:height(T)
        fprintf(fid, '| %s | %.5g | %.5g | %.5g | %.5g | %.5g | %.5g | %.4g | %.4g | %.4g |\n', ...
            T.method(i), T.f1_Hz(i), T.fu_Hz(i), T.fzero_Hz(i), ...
            T.Cp_pF(i), T.Cs_pF(i), T.Cps_pF(i), ...
            T.errPct_Cp(i), T.errPct_Cs(i), T.errPct_Cps(i));
    end
    fprintf(fid, '\n## 特征频率误差\n\n');
    fprintf(fid, '| method | feature | target Hz | estimated Hz | err %% |\n');
    fprintf(fid, '|---|---|---:|---:|---:|\n');
    for i = 1:height(F)
        fprintf(fid, '| %s | %s | %.5g | %.5g | %.4g |\n', ...
            F.method(i), F.feature(i), F.target_Hz(i), F.estimated_Hz(i), F.errPct(i));
    end
    fprintf(fid, '\n## 初步判断\n\n');
    fprintf(fid, '该实验检验了论文离线 TC 方法能否通过时域激励重构。若 fzero 附近的频谱能量不足，则 Cps 会首先退化；若 f1 分辨率或低频能量不足，则 Cp 会明显漂移。\n');
    fclose(fid);
end
