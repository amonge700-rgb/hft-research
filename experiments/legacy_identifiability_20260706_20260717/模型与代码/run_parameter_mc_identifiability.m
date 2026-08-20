% Monte Carlo parameter-identification experiment for PWM-driven HFT response.
% Migrated ideas:
%   1) Observability-aware active calibration: maximize lambda_min(FIM).
%   2) Bayesian optimal experiment design: average FIM over a parameter prior.
%   3) Robust regression: Huber loss for noisy frequency-response samples.

clear; clc; close all;
rng(20260707);

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
dataDir = fullfile(projectDir, char([25968 25454])); % Chinese folder name: data
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

% Frequency grid and true parameters.
Ns = 360;
f = logspace(3, 7, Ns);
w = 2*pi*f;

thetaTrue = [2e-6, 200e-12, 0.2]; % [Lsigma, Cp, Rac]
paramNames = {'Lsigma','Cp','Rac'};

cases = define_pwm_cases();
nCases = numel(cases);

% Noise model for a frequency-response estimate.
% Lower PWM spectral weight means poorer SNR at that frequency.
baseRelNoise = 0.006;
noiseFloor = 0.03;
maxRelNoise = 0.12;
outlierProb = 0.01;
outlierScale = 8;

% Estimation settings.
nTrials = 80;
logThetaTrue = log(thetaTrue);
startOffsets = [
     0.00,  0.00,  0.00
     0.15, -0.10,  0.10
    -0.15,  0.10, -0.10
     0.30,  0.20, -0.20
    -0.30, -0.20,  0.20
];
useHuber = true;

allThetaHat = nan(nTrials, 3, nCases);
allCost = nan(nTrials, nCases);
predicted = struct([]);

for iCase = 1:nCases
    U2 = pwm_spectrum_weight(f, cases(iCase).fs, cases(iCase).duty, cases(iCase).tr);
    relSigma = min(baseRelNoise ./ sqrt(noiseFloor + U2), maxRelNoise);
    Ztrue = hft_impedance(w, thetaTrue);
    measStd = relSigma .* max(abs(Ztrue), 1e-9);

    predicted(iCase).caseName = cases(iCase).name; %#ok<SAGROW>
    predicted(iCase).fimNominal = local_fim(w, thetaTrue, measStd);
    predicted(iCase).fimPriorAvg = prior_averaged_fim(w, thetaTrue, relSigma, 60);

    for kTrial = 1:nTrials
        noise = (randn(size(Ztrue)) + 1i*randn(size(Ztrue))) ./ sqrt(2) .* measStd;
        outlierMask = rand(size(Ztrue)) < outlierProb;
        noise(outlierMask) = noise(outlierMask) * outlierScale;
        Zmeas = Ztrue + noise;

        bestCost = inf;
        bestLogTheta = logThetaTrue;
        for iStart = 1:size(startOffsets,1)
            x0 = logThetaTrue + startOffsets(iStart,:);
            objective = @(x) parameter_objective(x, w, Zmeas, measStd, useHuber);
            opts = optimset('Display','off', 'MaxIter', 700, 'MaxFunEvals', 2500, ...
                'TolX', 1e-8, 'TolFun', 1e-8);
            [xHat, cost] = fminsearch(objective, x0, opts);
            if cost < bestCost
                bestCost = cost;
                bestLogTheta = xHat;
            end
        end

        allThetaHat(kTrial,:,iCase) = exp(bestLogTheta);
        allCost(kTrial,iCase) = bestCost;
    end
end

summary = build_summary_table(cases, allThetaHat, allCost, thetaTrue, predicted, paramNames);
summaryPath = fullfile(dataDir, 'mc_parameter_identifiability_summary.csv');
writetable(summary, summaryPath);

save(fullfile(dataDir, 'mc_parameter_identifiability.mat'), ...
    'summary', 'allThetaHat', 'allCost', 'thetaTrue', 'cases', 'predicted', ...
    'f', 'baseRelNoise', 'noiseFloor', 'maxRelNoise', 'nTrials');

write_markdown_report(summary, dataDir, nTrials, baseRelNoise, noiseFloor, maxRelNoise);
plot_mc_results(summary, allThetaHat, thetaTrue, cases, dataDir);

disp(summary);
disp(['Generated ', summaryPath]);

function cases = define_pwm_cases()
    cases(1).name = "best_tr50ns_fs500k_D03";
    cases(1).tr = 50e-9; cases(1).fs = 500e3; cases(1).duty = 0.3;

    cases(2).name = "good_tr100ns_fs500k_D03";
    cases(2).tr = 100e-9; cases(2).fs = 500e3; cases(2).duty = 0.3;

    cases(3).name = "mid_tr50ns_fs100k_D05";
    cases(3).tr = 50e-9; cases(3).fs = 100e3; cases(3).duty = 0.5;

    cases(4).name = "weak_tr500ns_fs50k_D07";
    cases(4).tr = 500e-9; cases(4).fs = 50e3; cases(4).duty = 0.7;
end

function Z = hft_impedance(w, theta)
    Lsigma = theta(1);
    Cp = theta(2);
    Rac = theta(3);
    Zs = Rac + 1i*w*Lsigma;
    Yp = 1i*w*Cp;
    Z = 1 ./ (1./Zs + Yp);
end

function U2 = pwm_spectrum_weight(f, fs, duty, tr)
    x = f / fs;
    dutyEnvelope = abs(sin(pi*duty*x) ./ max(pi*x, 1e-12));
    riseEnvelope = abs(sinc_local(f*tr));
    U2 = (dutyEnvelope .* riseEnvelope).^2;
    U2 = U2(:).' / max(U2(:));
end

function y = sinc_local(x)
    y = ones(size(x));
    nz = abs(x) > 1e-12;
    y(nz) = sin(pi*x(nz)) ./ (pi*x(nz));
end

function cost = parameter_objective(logTheta, w, Zmeas, measStd, useHuber)
    theta = exp(logTheta);
    Zhat = hft_impedance(w, theta);
    r = [(real(Zhat - Zmeas) ./ measStd), (imag(Zhat - Zmeas) ./ measStd)];
    if useHuber
        cost = sum(huber_loss(r(:), 1.5));
    else
        cost = sum(r(:).^2);
    end

    % Soft prior keeps unphysical excursions from dominating fminsearch.
    priorScale = [0.8, 0.8, 0.8];
    cost = cost + 0.02 * sum((logTheta - log([2e-6, 200e-12, 0.2])) ./ priorScale).^2;
end

function rho = huber_loss(r, delta)
    a = abs(r);
    rho = 0.5 * r.^2;
    idx = a > delta;
    rho(idx) = delta * (a(idx) - 0.5*delta);
end

function I = local_fim(w, theta, measStd)
    S = normalized_sensitivity(w, theta);
    W = 1 ./ max(measStd(:).^2, 1e-24);
    I = real(S' * (S .* W));
    I = I + 1e-12 * eye(size(I));
end

function Iavg = prior_averaged_fim(w, thetaCenter, relSigma, nPrior)
    Iavg = zeros(3,3);
    for k = 1:nPrior
        theta = thetaCenter .* exp(0.20 * randn(1,3));
        Z = hft_impedance(w, theta);
        measStd = relSigma .* max(abs(Z), 1e-9);
        Iavg = Iavg + local_fim(w, theta, measStd);
    end
    Iavg = Iavg / nPrior;
end

function S = normalized_sensitivity(w, theta)
    step = 1e-4;
    H0 = hft_impedance(w, theta);
    S = zeros(numel(w), 3);
    for i = 1:3
        tp = theta; tm = theta;
        tp(i) = theta(i) * exp(step);
        tm(i) = theta(i) * exp(-step);
        Hp = hft_impedance(w, tp);
        Hm = hft_impedance(w, tm);
        S(:,i) = ((Hp - Hm) / (2*step)).';
    end
    S = S ./ max(abs(H0(:)), 1e-9);
end

function summary = build_summary_table(cases, allThetaHat, allCost, thetaTrue, predicted, paramNames)
    nCases = numel(cases);
    rows = struct([]);
    for iCase = 1:nCases
        thetaHat = squeeze(allThetaHat(:,:,iCase));
        relErr = (thetaHat - thetaTrue) ./ thetaTrue;
        I = predicted(iCase).fimNominal;
        Iprior = predicted(iCase).fimPriorAvg;

        rows(iCase).caseName = string(cases(iCase).name); %#ok<AGROW>
        rows(iCase).tr_ns = cases(iCase).tr * 1e9;
        rows(iCase).fs_kHz = cases(iCase).fs / 1e3;
        rows(iCase).duty = cases(iCase).duty;
        rows(iCase).lambdaMinNominal = min(eig(I));
        rows(iCase).condNominal = cond(I);
        rows(iCase).detNominal = det(I);
        rows(iCase).lambdaMinPriorAvg = min(eig(Iprior));
        rows(iCase).condPriorAvg = cond(Iprior);
        rows(iCase).meanCost = mean(allCost(:,iCase));

        for p = 1:numel(paramNames)
            rows(iCase).(['biasPct_', paramNames{p}]) = 100 * mean(relErr(:,p));
            rows(iCase).(['stdPct_', paramNames{p}]) = 100 * std(relErr(:,p));
            rows(iCase).(['rmsePct_', paramNames{p}]) = 100 * sqrt(mean(relErr(:,p).^2));
        end
        rows(iCase).meanRmsePct = mean([rows(iCase).rmsePct_Lsigma, rows(iCase).rmsePct_Cp, rows(iCase).rmsePct_Rac]);
    end
    summary = struct2table(rows);
    summary = sortrows(summary, 'meanRmsePct', 'ascend');
end

function write_markdown_report(summary, dataDir, nTrials, baseRelNoise, noiseFloor, maxRelNoise)
    fid = fopen(fullfile(dataDir, 'mc_parameter_identifiability_report.md'), 'w');
    fprintf(fid, '# Monte Carlo 参数可辨识性实验\n\n');
    fprintf(fid, '生成时间：%s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, '## 设置\n\n');
    fprintf(fid, '- 重复次数：%d\n', nTrials);
    fprintf(fid, '- 基础相对噪声：%.3g\n', baseRelNoise);
    fprintf(fid, '- PWM 权重噪声底：%.3g\n', noiseFloor);
    fprintf(fid, '- 最大相对噪声：%.3g\n', maxRelNoise);
    fprintf(fid, '- 参数：`[Lsigma, Cp, Rac]`\n');
    fprintf(fid, '- 方法：多起点 `fminsearch` + Huber robust loss + 先验平均 Fisher 信息\n\n');

    fprintf(fid, '## 主要结果\n\n');
    fprintf(fid, '| case | meanRmsePct | std Lsigma %% | std Cp %% | std Rac %% | lambdaMinPriorAvg | condPriorAvg |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|\n');
    for k = 1:height(summary)
        fprintf(fid, '| %s | %.4g | %.4g | %.4g | %.4g | %.4e | %.4e |\n', ...
            summary.caseName(k), summary.meanRmsePct(k), ...
            summary.stdPct_Lsigma(k), summary.stdPct_Cp(k), summary.stdPct_Rac(k), ...
            summary.lambdaMinPriorAvg(k), summary.condPriorAvg(k));
    end

    fprintf(fid, '\n## 解释\n\n');
    fprintf(fid, '这个实验直接比较参数反演误差，而不是只比较频响拟合误差。');
    fprintf(fid, '它更接近在线宽频辨识项目的核心目标：在不同 PWM 条件下，判断 `Lsigma`, `Cp`, `Rac` 的估计方差是否可接受。\n\n');
    fprintf(fid, '本实验迁移了机器人主动标定中的 `lambda_min(FIM)` 指标，');
    fprintf(fid, '并加入类似 Bayesian optimal design 的先验平均 Fisher 信息，');
    fprintf(fid, '避免只在单一标称参数点上得出过强结论。\n');
    fclose(fid);
end

function plot_mc_results(summary, allThetaHat, thetaTrue, cases, dataDir)
    caseNames = string(summary.caseName);
    originalNames = string({cases.name});
    [~, order] = ismember(caseNames, originalNames);

    figure('Color', 'w', 'Position', [100, 100, 980, 420]);
    bar(1:height(summary), summary.meanRmsePct);
    set(gca, 'XTick', 1:height(summary), 'XTickLabel', cellstr(caseNames));
    xtickangle(25);
    ylabel('Mean parameter RMSE (%)');
    title('Monte Carlo parameter-estimation error');
    grid on;
    saveas(gcf, fullfile(dataDir, 'mc_parameter_rmse_bar.png'));

    paramLabels = {'Lsigma','Cp','Rac'};
    figure('Color', 'w', 'Position', [100, 100, 1100, 720]);
    for p = 1:3
        subplot(1,3,p);
        for k = 1:numel(order)
            thetaHat = squeeze(allThetaHat(:,p,order(k)));
            relErrPct = 100 * (thetaHat - thetaTrue(p)) / thetaTrue(p);
            x = k + 0.16 * randn(size(relErrPct));
            plot(x, relErrPct, '.', 'MarkerSize', 8); hold on;
            q = prctile_local(relErrPct, [25 50 75]);
            plot([k-0.22, k+0.22], [q(2), q(2)], 'k-', 'LineWidth', 1.5);
            plot([k, k], [q(1), q(3)], 'k-', 'LineWidth', 1.2);
        end
        set(gca, 'XTick', 1:numel(order), 'XTickLabel', cellstr(caseNames));
        ylabel('Relative error (%)');
        title(paramLabels{p});
        grid on;
        xtickangle(30);
    end
    saveas(gcf, fullfile(dataDir, 'mc_parameter_error_boxplot.png'));

    figure('Color', 'w', 'Position', [100, 100, 980, 420]);
    yyaxis left;
    semilogy(1:height(summary), summary.lambdaMinPriorAvg, 'o-', 'LineWidth', 1.4);
    ylabel('Prior-averaged lambda min');
    yyaxis right;
    plot(1:height(summary), summary.meanRmsePct, 's-', 'LineWidth', 1.4);
    ylabel('Mean parameter RMSE (%)');
    set(gca, 'XTick', 1:height(summary), 'XTickLabel', caseNames);
    xtickangle(25);
    grid on;
    title('FIM prediction versus Monte Carlo parameter error');
    saveas(gcf, fullfile(dataDir, 'mc_fim_vs_rmse.png'));
end

function q = prctile_local(x, p)
    x = sort(x(:));
    n = numel(x);
    q = zeros(size(p));
    for i = 1:numel(p)
        pos = 1 + (n-1) * p(i) / 100;
        lo = floor(pos);
        hi = ceil(pos);
        if lo == hi
            q(i) = x(lo);
        else
            q(i) = x(lo) + (pos-lo) * (x(hi)-x(lo));
        end
    end
end
