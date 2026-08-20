% Multi-port frequency optimal experimental design demo.
% Goal:
%   Test whether E-optimal frequency selection improves multi-parameter
%   identification for a two-port HFT admittance model.
%
% Migrated method:
%   E-optimal design from impedance spectroscopy / optimal experiment design.
%   We greedily select frequencies that maximize lambda_min(FIM).

clear; clc; close all;
rng(20260707);

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
dataDir = fullfile(projectDir, char([25968 25454])); % Chinese folder name: data
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

paramNames = {'Lsigma','Cps','Cg','Rac','Gg'};
thetaTrue = [2.0e-6, 80e-12, 120e-12, 0.20, 1e-6];
channels = {'Y11','Y12','Y22'};

% Candidate frequencies for design.
nCandidate = 260;
fCand = logspace(3, 7, nCandidate);
wCand = 2*pi*fCand;

% Use the best PWM condition found previously.
pwm.tr = 50e-9;
pwm.fs = 500e3;
pwm.duty = 0.3;
U2 = pwm_spectrum_weight(fCand, pwm.fs, pwm.duty, pwm.tr);

baseRelNoise = 0.006;
noiseFloor = 0.03;
relSigma = min(baseRelNoise ./ sqrt(noiseFloor + U2), 0.12);

% Build single-frequency FIM contributions.
contribNominal = fim_contributions_by_frequency(wCand, thetaTrue, channels, relSigma);
contribPrior = prior_averaged_fim_contributions(wCand, thetaTrue, channels, relSigma, 40);

Klist = [20, 40, 80];
maxK = max(Klist);
idxEoptNominalSeq = greedy_eopt_select(contribNominal, maxK);
idxEoptPriorSeq = greedy_eopt_select(contribPrior, maxK);

rows = [];
rowIndex = 0;
for iK = 1:numel(Klist)
    K = Klist(iK);
    strategies = define_strategies(nCandidate, K, idxEoptNominalSeq, idxEoptPriorSeq);
    for iStrat = 1:numel(strategies)
        idx = strategies(iStrat).idx;
        Itrue = sum_fim(contribNominal, idx);
        Iprior = sum_fim(contribPrior, idx);

        rowIndex = rowIndex + 1;
        rows(rowIndex).strategy = string(strategies(iStrat).name); %#ok<SAGROW>
        rows(rowIndex).K = K;
        rows(rowIndex).lambdaMinNominal = min(eig(Itrue));
        rows(rowIndex).condNominal = cond(Itrue);
        rows(rowIndex).detNominal = det(Itrue);
        rows(rowIndex).lambdaMinPrior = min(eig(Iprior));
        rows(rowIndex).condPrior = cond(Iprior);
        rows(rowIndex).fMinHz = min(fCand(idx));
        rows(rowIndex).fMaxHz = max(fCand(idx));
    end
end

Tdesign = struct2table(rows);
Tdesign = sortrows(Tdesign, {'K','lambdaMinPrior'}, {'ascend','descend'});
writetable(Tdesign, fullfile(dataDir, 'oed_frequency_design_summary.csv'));

% Monte Carlo verification for K=40 plus full candidate set.
nTrials = 35;
Kmc = 40;
mcStrategies = define_strategies(nCandidate, Kmc, idxEoptNominalSeq, idxEoptPriorSeq);
mcStrategies(end+1).name = "full_260_freq";
mcStrategies(end).idx = 1:nCandidate;

Tmc = run_mc_verification(fCand, thetaTrue, channels, relSigma, mcStrategies, nTrials, paramNames);
writetable(Tmc, fullfile(dataDir, 'oed_frequency_mc_summary.csv'));

save(fullfile(dataDir, 'oed_frequency_selection_demo.mat'), ...
    'Tdesign', 'Tmc', 'fCand', 'idxEoptNominalSeq', 'idxEoptPriorSeq', ...
    'thetaTrue', 'channels', 'relSigma', 'pwm', 'Klist');

plot_oed_results(Tdesign, Tmc, fCand, idxEoptNominalSeq, idxEoptPriorSeq, dataDir);
write_ascii_report(Tdesign, Tmc, dataDir);

disp(Tdesign);
disp(Tmc);

function strategies = define_strategies(nCandidate, K, idxEoptNominalSeq, idxEoptPriorSeq)
    strategies(1).name = "log_uniform";
    strategies(1).idx = unique(round(linspace(1, nCandidate, K)));
    strategies(1).idx = pad_to_k(strategies(1).idx, nCandidate, K);

    strategies(2).name = "eopt_nominal";
    strategies(2).idx = sort(idxEoptNominalSeq(1:K));

    strategies(3).name = "eopt_prior_avg";
    strategies(3).idx = sort(idxEoptPriorSeq(1:K));
end

function idx = pad_to_k(idx, nCandidate, K)
    if numel(idx) >= K
        idx = idx(1:K);
        return;
    end
    extra = setdiff(1:nCandidate, idx, 'stable');
    idx = sort([idx(:); extra(1:(K-numel(idx))).']);
end

function contrib = fim_contributions_by_frequency(w, theta, channels, relSigma)
    nFreq = numel(w);
    nParam = numel(theta);
    contrib = zeros(nParam, nParam, nFreq);
    for k = 1:nFreq
        S = multiport_sensitivity(w(k), theta, channels);
        H = select_channels(hft_twoport_y(w(k), theta), channels);
        sigma = relSigma(k) * max(abs(H(:)), 1e-12);
        W = repmat(1 ./ sigma(:).^2, 1, nParam);
        Sk = reshape(S, [], nParam);
        contrib(:,:,k) = real(Sk' * (Sk .* W));
    end
end

function contribAvg = prior_averaged_fim_contributions(w, thetaCenter, channels, relSigma, nPrior)
    contribAvg = zeros(numel(thetaCenter), numel(thetaCenter), numel(w));
    for p = 1:nPrior
        theta = thetaCenter .* exp(0.22 * randn(size(thetaCenter)));
        contribAvg = contribAvg + fim_contributions_by_frequency(w, theta, channels, relSigma);
    end
    contribAvg = contribAvg / nPrior;
end

function selected = greedy_eopt_select(contrib, K)
    nFreq = size(contrib, 3);
    selected = zeros(1, K);
    current = 1e-18 * eye(size(contrib,1));
    available = true(1, nFreq);

    for k = 1:K
        bestScore = -inf;
        bestIdx = 1;
        for i = find(available)
            candidate = current + contrib(:,:,i);
            score = min(eig(candidate));
            if score > bestScore
                bestScore = score;
                bestIdx = i;
            end
        end
        selected(k) = bestIdx;
        current = current + contrib(:,:,bestIdx);
        available(bestIdx) = false;
    end
end

function I = sum_fim(contrib, idx)
    I = zeros(size(contrib,1), size(contrib,2));
    for k = 1:numel(idx)
        I = I + contrib(:,:,idx(k));
    end
    I = I + 1e-18 * eye(size(I));
end

function Tmc = run_mc_verification(fCand, thetaTrue, channels, relSigma, strategies, nTrials, paramNames)
    rows = [];
    wCand = 2*pi*fCand;
    for iStrat = 1:numel(strategies)
        idx = strategies(iStrat).idx;
        w = wCand(idx);
        sigmaRel = relSigma(idx);
        thetaHat = nan(nTrials, numel(thetaTrue));
        cost = nan(nTrials, 1);

        Htrue = select_channels(hft_twoport_y(w, thetaTrue), channels);
        measStd = sigmaRel(:).' .* max(abs(Htrue), 1e-12);

        for t = 1:nTrials
            noise = (randn(size(Htrue)) + 1i*randn(size(Htrue))) ./ sqrt(2) .* measStd;
            Hmeas = Htrue + noise;
            [thetaHat(t,:), cost(t)] = estimate_theta(w, Hmeas, measStd, thetaTrue, channels);
        end

        relErr = (thetaHat - thetaTrue) ./ thetaTrue;
        rows(iStrat).strategy = string(strategies(iStrat).name); %#ok<AGROW>
        rows(iStrat).K = numel(idx);
        rows(iStrat).meanCost = mean(cost);
        for p = 1:numel(paramNames)
            rows(iStrat).(['biasPct_', paramNames{p}]) = 100 * mean(relErr(:,p));
            rows(iStrat).(['stdPct_', paramNames{p}]) = 100 * std(relErr(:,p));
            rows(iStrat).(['rmsePct_', paramNames{p}]) = 100 * sqrt(mean(relErr(:,p).^2));
        end
        rows(iStrat).meanRmsePct = mean([rows(iStrat).rmsePct_Lsigma, ...
            rows(iStrat).rmsePct_Cps, rows(iStrat).rmsePct_Cg, ...
            rows(iStrat).rmsePct_Rac, rows(iStrat).rmsePct_Gg]);
    end
    Tmc = struct2table(rows);
    Tmc = sortrows(Tmc, 'meanRmsePct', 'ascend');
end

function [thetaBest, costBest] = estimate_theta(w, Hmeas, measStd, thetaTrue, channels)
    logTrue = log(thetaTrue);
    starts = [
         0,     0,     0,     0,     0
         0.10, -0.10,  0.08,  0.08, -0.12
        -0.12,  0.10, -0.08, -0.08,  0.12
         0.22,  0.18, -0.16,  0.12, -0.18
    ];
    costBest = inf;
    thetaBest = thetaTrue;
    opts = optimset('Display','off', 'MaxIter', 800, 'MaxFunEvals', 3000, ...
        'TolX', 1e-8, 'TolFun', 1e-8);
    for i = 1:size(starts,1)
        x0 = logTrue + starts(i,:);
        obj = @(x) objective(x, w, Hmeas, measStd, thetaTrue, channels);
        [x, cost] = fminsearch(obj, x0, opts);
        if cost < costBest
            costBest = cost;
            thetaBest = exp(x);
        end
    end
end

function cost = objective(logTheta, w, Hmeas, measStd, thetaPrior, channels)
    theta = exp(logTheta);
    Hhat = select_channels(hft_twoport_y(w, theta), channels);
    r = [(real(Hhat-Hmeas) ./ measStd), (imag(Hhat-Hmeas) ./ measStd)];
    cost = sum(huber_loss(r(:), 1.5));
    cost = cost + 0.01 * sum(((logTheta - log(thetaPrior)) ./ 0.9).^2);
end

function rho = huber_loss(r, delta)
    a = abs(r);
    rho = 0.5 * r.^2;
    idx = a > delta;
    rho(idx) = delta * (a(idx) - 0.5*delta);
end

function Y = hft_twoport_y(w, theta)
    Lsigma = theta(1);
    Cps = theta(2);
    Cg = theta(3);
    Rac = theta(4);
    Gg = theta(5);
    s = 1i*w(:).';
    ys = 1 ./ (Rac + s*Lsigma);
    yps = s * Cps;
    yg = Gg + s * Cg;

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
    H0 = select_channels(hft_twoport_y(w, theta), channels);
    nParam = numel(theta);
    S = zeros(numel(H0), nParam);
    step = 1e-4;
    for p = 1:nParam
        tp = theta; tm = theta;
        tp(p) = theta(p) * exp(step);
        tm(p) = theta(p) * exp(-step);
        Hp = select_channels(hft_twoport_y(w, tp), channels);
        Hm = select_channels(hft_twoport_y(w, tm), channels);
        dH = (Hp - Hm) / (2*step);
        S(:,p) = reshape(dH, [], 1);
    end
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

function plot_oed_results(Tdesign, Tmc, fCand, idxNominal, idxPrior, dataDir)
    figure('Color','w','Position',[100,100,980,420]);
    Kvals = unique(Tdesign.K, 'stable');
    names = unique(Tdesign.strategy, 'stable');
    for i = 1:numel(names)
        y = nan(size(Kvals));
        for k = 1:numel(Kvals)
            idx = Tdesign.K == Kvals(k) & Tdesign.strategy == names(i);
            y(k) = Tdesign.lambdaMinPrior(idx);
        end
        semilogy(Kvals, y, 'o-', 'LineWidth', 1.4); hold on;
    end
    grid on;
    xlabel('Number of selected frequencies');
    ylabel('Prior-averaged lambda min');
    legend(cellstr(names), 'Location', 'best');
    title('E-optimal frequency selection improves FIM');
    saveas(gcf, fullfile(dataDir, 'oed_frequency_lambda_min.png'));

    figure('Color','w','Position',[100,100,980,420]);
    bar(1:height(Tmc), Tmc.meanRmsePct);
    set(gca, 'XTick', 1:height(Tmc), 'XTickLabel', cellstr(Tmc.strategy));
    xtickangle(20);
    ylabel('Mean parameter RMSE (%)');
    title('Monte Carlo verification of frequency selection');
    grid on;
    saveas(gcf, fullfile(dataDir, 'oed_frequency_mc_rmse.png'));

    figure('Color','w','Position',[100,100,980,420]);
    semilogx(fCand(idxNominal(1:80)), ones(1,80), 'o', 'MarkerSize', 5); hold on;
    semilogx(fCand(idxPrior(1:80)), 1.08*ones(1,80), 'x', 'MarkerSize', 5);
    ylim([0.92, 1.16]);
    set(gca, 'YTick', [1, 1.08], 'YTickLabel', {'nominal', 'prior avg'});
    grid on;
    xlabel('Selected frequency (Hz)');
    title('First 80 E-optimal selected frequencies');
    saveas(gcf, fullfile(dataDir, 'oed_frequency_selected_points.png'));
end

function write_ascii_report(Tdesign, Tmc, dataDir)
    fid = fopen(fullfile(dataDir, 'oed_frequency_selection_report.txt'), 'w');
    fprintf(fid, 'OED frequency-selection demo for multi-port HFT identification\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, 'Design summary: see oed_frequency_design_summary.csv\n');
    fprintf(fid, 'Monte Carlo summary: see oed_frequency_mc_summary.csv\n\n');
    fprintf(fid, 'Best design rows by K using prior-averaged lambda_min:\n');
    Kvals = unique(Tdesign.K, 'stable');
    for i = 1:numel(Kvals)
        idx = find(Tdesign.K == Kvals(i), 1, 'first');
        fprintf(fid, 'K=%d: %s, lambdaMinPrior=%.6e, condPrior=%.6e\n', ...
            Tdesign.K(idx), Tdesign.strategy(idx), ...
            Tdesign.lambdaMinPrior(idx), Tdesign.condPrior(idx));
    end
    fprintf(fid, '\nMonte Carlo ranking:\n');
    for i = 1:height(Tmc)
        fprintf(fid, '%s, K=%d, meanRmsePct=%.6e\n', ...
            Tmc.strategy(i), Tmc.K(i), Tmc.meanRmsePct(i));
    end
    fclose(fid);
end
