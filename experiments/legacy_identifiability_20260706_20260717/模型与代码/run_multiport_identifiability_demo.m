% Multi-port HFT identifiability demo.
% Purpose:
%   Compare parameter identifiability using single-port and two-port
%   admittance observations under PWM/steep-pulse excitation.
%
% Model:
%   Two-port admittance matrix with leakage branch, inter-winding
%   capacitance, ground capacitance, AC resistance, and dielectric loss.
%
% Parameters:
%   theta = [Lsigma, Cps, Cg, Rac, Gg]

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
dataDir = fullfile(projectDir, char([25968 25454])); % Chinese folder name: data
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

f = logspace(3, 7, 600);
w = 2*pi*f;

theta.Lsigma = 2.0e-6;     % leakage inductance, H
theta.Cps = 80e-12;        % inter-winding capacitance, F
theta.Cg = 120e-12;        % port-to-ground capacitance per port, F
theta.Rac = 0.20;          % AC copper loss, Ohm
theta.Gg = 1e-6;           % dielectric/conductive loss to ground, S

paramNames = {'Lsigma','Cps','Cg','Rac','Gg'};

pwmCases = define_pwm_cases();
measurementSets = define_measurement_sets();

rows = [];
rowIndex = 0;
for iCase = 1:numel(pwmCases)
    U2 = pwm_spectrum_weight(f, pwmCases(iCase).fs, pwmCases(iCase).duty, pwmCases(iCase).tr);
    for iMeas = 1:numel(measurementSets)
        S = multiport_sensitivity(w, theta, measurementSets(iMeas).channels);
        I = fisher_information(S, U2);
        eigvals = eig(I);
        covApprox = pinv(I);

        rowIndex = rowIndex + 1;
        rows(rowIndex).pwmCase = string(pwmCases(iCase).name); %#ok<SAGROW>
        rows(rowIndex).measurementSet = string(measurementSets(iMeas).name);
        rows(rowIndex).nChannels = numel(measurementSets(iMeas).channels);
        rows(rowIndex).tr_ns = pwmCases(iCase).tr * 1e9;
        rows(rowIndex).fs_kHz = pwmCases(iCase).fs / 1e3;
        rows(rowIndex).duty = pwmCases(iCase).duty;
        rows(rowIndex).rankI = rank(I, 1e-8);
        rows(rowIndex).lambdaMin = min(eigvals);
        rows(rowIndex).condI = cond(I);
        rows(rowIndex).detI = det(I);
        rows(rowIndex).traceInvI = trace(covApprox);
        for p = 1:numel(paramNames)
            rows(rowIndex).(['crlb_', paramNames{p}]) = covApprox(p,p);
        end
    end
end

T = struct2table(rows);
T = sortrows(T, {'measurementSet','condI'}, {'ascend','ascend'});

summaryPath = fullfile(dataDir, 'multiport_identifiability_summary.csv');
writetable(T, summaryPath);

write_report(T, dataDir);
plot_multiport_results(T, dataDir);

disp(T);
disp(['Generated ', summaryPath]);

function cases = define_pwm_cases()
    cases(1).name = "best_50ns_500k_D03";
    cases(1).tr = 50e-9; cases(1).fs = 500e3; cases(1).duty = 0.3;

    cases(2).name = "mid_50ns_100k_D05";
    cases(2).tr = 50e-9; cases(2).fs = 100e3; cases(2).duty = 0.5;

    cases(3).name = "weak_500ns_50k_D07";
    cases(3).tr = 500e-9; cases(3).fs = 50e3; cases(3).duty = 0.7;
end

function sets = define_measurement_sets()
    sets(1).name = "single_Y11";
    sets(1).channels = {'Y11'};

    sets(2).name = "two_diag_Y11_Y22";
    sets(2).channels = {'Y11','Y22'};

    sets(3).name = "twoport_Y11_Y12_Y22";
    sets(3).channels = {'Y11','Y12','Y22'};
end

function Y = hft_twoport_y(w, theta)
    s = 1i*w(:).';
    ys = 1 ./ (theta.Rac + s*theta.Lsigma);
    yps = s * theta.Cps;
    yg = theta.Gg + s * theta.Cg;

    Y11 = yg + yps + ys;
    Y22 = yg + yps + ys;
    Y12 = -yps - ys;

    Y = zeros(2,2,numel(w));
    Y(1,1,:) = Y11;
    Y(2,2,:) = Y22;
    Y(1,2,:) = Y12;
    Y(2,1,:) = Y12;
end

function H = select_channels(Y, channels)
    nFreq = size(Y,3);
    H = zeros(numel(channels), nFreq);
    for k = 1:numel(channels)
        switch channels{k}
            case 'Y11'
                H(k,:) = reshape(Y(1,1,:), 1, nFreq);
            case 'Y12'
                H(k,:) = reshape(Y(1,2,:), 1, nFreq);
            case 'Y22'
                H(k,:) = reshape(Y(2,2,:), 1, nFreq);
        end
    end
end

function S = multiport_sensitivity(w, theta, channels)
    names = {'Lsigma','Cps','Cg','Rac','Gg'};
    Y0 = hft_twoport_y(w, theta);
    H0 = select_channels(Y0, channels);
    nObs = numel(H0);
    S = zeros(nObs, numel(names));
    step = 1e-4;

    for p = 1:numel(names)
        tp = theta; tm = theta;
        name = names{p};
        tp.(name) = theta.(name) * exp(step);
        tm.(name) = theta.(name) * exp(-step);

        Hp = select_channels(hft_twoport_y(w, tp), channels);
        Hm = select_channels(hft_twoport_y(w, tm), channels);
        dH = (Hp - Hm) / (2*step);
        scale = max(abs(H0), 1e-12);
        S(:,p) = reshape(dH ./ scale, [], 1);
    end
end

function U2 = pwm_spectrum_weight(f, fs, duty, tr)
    x = f / fs;
    dutyEnvelope = abs(sin(pi*duty*x) ./ max(pi*x, 1e-12));
    riseEnvelope = abs(sinc_local(f*tr));
    U2 = (dutyEnvelope .* riseEnvelope).^2;
    U2 = U2(:) / max(U2(:));
end

function y = sinc_local(x)
    y = ones(size(x));
    nz = abs(x) > 1e-12;
    y(nz) = sin(pi*x(nz)) ./ (pi*x(nz));
end

function I = fisher_information(S, U2)
    nChannels = size(S,1) / numel(U2);
    W = repmat(U2(:), nChannels, 1);
    I = real(S' * (S .* W));
    I = I + 1e-12 * eye(size(I));
end

function write_report(T, dataDir)
    reportPath = fullfile(dataDir, 'multiport_identifiability_report.md');
    fid = fopen(reportPath, 'w');
    fprintf(fid, '# 多端口可辨识性实验\n\n');
    fprintf(fid, '生成时间：%s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, '## 模型\n\n');
    fprintf(fid, '两端口高频变压器导纳矩阵，参数为：`[Lsigma, Cps, Cg, Rac, Gg]`。\n\n');
    fprintf(fid, '比较三种测量集合：\n\n');
    fprintf(fid, '- `single_Y11`：只测输入端口导纳；\n');
    fprintf(fid, '- `two_diag_Y11_Y22`：测两个端口自导纳；\n');
    fprintf(fid, '- `twoport_Y11_Y12_Y22`：测完整对称两端口导纳的独立元素。\n\n');
    fprintf(fid, '## 最优结果摘录\n\n');
    fprintf(fid, '| measurementSet | pwmCase | rankI | lambdaMin | condI | traceInvI |\n');
    fprintf(fid, '|---|---|---:|---:|---:|---:|\n');
    sets = unique(T.measurementSet, 'stable');
    for k = 1:numel(sets)
        idx = find(T.measurementSet == sets(k), 1, 'first');
        fprintf(fid, '| %s | %s | %d | %.4e | %.4e | %.4e |\n', ...
            T.measurementSet(idx), T.pwmCase(idx), T.rankI(idx), ...
            T.lambdaMin(idx), T.condI(idx), T.traceInvI(idx));
    end
    fprintf(fid, '\n## 解释\n\n');
    fprintf(fid, '这个实验用于验证多端口测量是否能显著提升寄生参数可辨识性。');
    fprintf(fid, '如果 `twoport_Y11_Y12_Y22` 的 rank 更高、condition number 更低，');
    fprintf(fid, '说明后续实验应优先采集多端口响应，而不是只看单端口输入阻抗。\n');
    fclose(fid);
end

function plot_multiport_results(T, dataDir)
    sets = unique(T.measurementSet, 'stable');

    bestRows = zeros(numel(sets), 1);
    for k = 1:numel(sets)
        idx = find(T.measurementSet == sets(k));
        [~, localBest] = min(T.condI(idx));
        bestRows(k) = idx(localBest);
    end

    figure('Color', 'w', 'Position', [100, 100, 980, 420]);
    bar(1:numel(bestRows), log10(T.condI(bestRows)));
    set(gca, 'XTick', 1:numel(bestRows), 'XTickLabel', cellstr(sets));
    xtickangle(20);
    ylabel('log10 cond(I), lower is better');
    title('Best-case identifiability by measurement set');
    grid on;
    saveas(gcf, fullfile(dataDir, 'multiport_best_cond_by_measurement.png'));

    figure('Color', 'w', 'Position', [100, 100, 980, 420]);
    bar(1:numel(bestRows), T.rankI(bestRows));
    set(gca, 'XTick', 1:numel(bestRows), 'XTickLabel', cellstr(sets));
    xtickangle(20);
    ylabel('rank(I)');
    title('FIM rank by measurement set');
    grid on;
    saveas(gcf, fullfile(dataDir, 'multiport_rank_by_measurement.png'));

    figure('Color', 'w', 'Position', [100, 100, 980, 420]);
    for k = 1:numel(sets)
        idx = find(T.measurementSet == sets(k));
        plot(1:numel(idx), log10(T.condI(idx)), 'o-', 'LineWidth', 1.3); hold on;
    end
    grid on;
    xlabel('PWM case index sorted within set');
    ylabel('log10 cond(I)');
    legend(cellstr(sets), 'Location', 'best');
    title('PWM condition effect under different measurement sets');
    saveas(gcf, fullfile(dataDir, 'multiport_pwm_vs_measurement_cond.png'));
end
