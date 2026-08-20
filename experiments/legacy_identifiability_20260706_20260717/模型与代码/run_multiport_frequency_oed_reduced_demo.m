% Reduced-parameter OED demo.
% Fix Gg after practical identifiability screening and estimate:
%   theta_est = [Lsigma, Cps, Cg, Rac]
% while keeping Gg fixed at its nominal value.

clear; clc; close all;
rng(20260708);

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
dataDir = fullfile(projectDir, char([25968 25454]));
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

paramNames = {'Lsigma','Cps','Cg','Rac'};
thetaTrue4 = [2.0e-6, 80e-12, 120e-12, 0.20];
GgFixed = 1e-6;
channels = {'Y11','Y12','Y22'};

nCandidate = 260;
fCand = logspace(3, 7, nCandidate);
wCand = 2*pi*fCand;

pwm.tr = 50e-9;
pwm.fs = 500e3;
pwm.duty = 0.3;
U2 = pwm_spectrum_weight(fCand, pwm.fs, pwm.duty, pwm.tr);
baseRelNoise = 0.006;
noiseFloor = 0.03;
relSigma = min(baseRelNoise ./ sqrt(noiseFloor + U2), 0.12);

contribPrior = prior_averaged_fim_contributions(wCand, thetaTrue4, GgFixed, channels, relSigma, 40);
contribNominal = fim_contributions_by_frequency(wCand, thetaTrue4, GgFixed, channels, relSigma);

Kmc = 40;
idxEoptPrior = greedy_eopt_select(contribPrior, Kmc);
idxEoptNominal = greedy_eopt_select(contribNominal, Kmc);
idxUniform = unique(round(linspace(1, nCandidate, Kmc)));

strategies(1).name = "log_uniform_4param";
strategies(1).idx = idxUniform;
strategies(2).name = "eopt_nominal_4param";
strategies(2).idx = sort(idxEoptNominal);
strategies(3).name = "eopt_prior_avg_4param";
strategies(3).idx = sort(idxEoptPrior);
strategies(4).name = "full_260_4param";
strategies(4).idx = 1:nCandidate;

nTrials = 35;
Tmc = run_mc_verification(fCand, thetaTrue4, GgFixed, channels, relSigma, strategies, nTrials, paramNames);
writetable(Tmc, fullfile(dataDir, 'oed_frequency_reduced_mc_summary.csv'));

save(fullfile(dataDir, 'oed_frequency_reduced_demo.mat'), ...
    'Tmc', 'fCand', 'idxEoptPrior', 'idxEoptNominal', 'idxUniform', ...
    'thetaTrue4', 'GgFixed', 'channels', 'relSigma');

plot_results(Tmc, fCand, idxEoptPrior, idxEoptNominal, idxUniform, dataDir);
write_report(Tmc, dataDir);
disp(Tmc);

function contribAvg = prior_averaged_fim_contributions(w, thetaCenter, GgFixed, channels, relSigma, nPrior)
    contribAvg = zeros(numel(thetaCenter), numel(thetaCenter), numel(w));
    for p = 1:nPrior
        theta = thetaCenter .* exp(0.22 * randn(size(thetaCenter)));
        contribAvg = contribAvg + fim_contributions_by_frequency(w, theta, GgFixed, channels, relSigma);
    end
    contribAvg = contribAvg / nPrior;
end

function contrib = fim_contributions_by_frequency(w, theta, GgFixed, channels, relSigma)
    nFreq = numel(w);
    nParam = numel(theta);
    contrib = zeros(nParam, nParam, nFreq);
    for k = 1:nFreq
        S = multiport_sensitivity(w(k), theta, GgFixed, channels);
        H = select_channels(hft_twoport_y(w(k), theta, GgFixed), channels);
        sigma = relSigma(k) * max(abs(H(:)), 1e-12);
        W = repmat(1 ./ sigma(:).^2, 1, nParam);
        contrib(:,:,k) = real(S' * (S .* W));
    end
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
            score = min(eig(current + contrib(:,:,i)));
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

function Tmc = run_mc_verification(fCand, thetaTrue, GgFixed, channels, relSigma, strategies, nTrials, paramNames)
    wCand = 2*pi*fCand;
    rows = [];
    for iStrat = 1:numel(strategies)
        idx = strategies(iStrat).idx;
        w = wCand(idx);
        sigmaRel = relSigma(idx);
        Htrue = select_channels(hft_twoport_y(w, thetaTrue, GgFixed), channels);
        measStd = sigmaRel(:).' .* max(abs(Htrue), 1e-12);
        thetaHat = nan(nTrials, numel(thetaTrue));
        for t = 1:nTrials
            noise = (randn(size(Htrue)) + 1i*randn(size(Htrue))) ./ sqrt(2) .* measStd;
            Hmeas = Htrue + noise;
            thetaHat(t,:) = estimate_theta(w, Hmeas, measStd, thetaTrue, GgFixed, channels);
        end
        relErr = (thetaHat - thetaTrue) ./ thetaTrue;
        rows(iStrat).strategy = string(strategies(iStrat).name); %#ok<AGROW>
        rows(iStrat).K = numel(idx);
        for p = 1:numel(paramNames)
            rows(iStrat).(['biasPct_', paramNames{p}]) = 100 * mean(relErr(:,p));
            rows(iStrat).(['stdPct_', paramNames{p}]) = 100 * std(relErr(:,p));
            rows(iStrat).(['rmsePct_', paramNames{p}]) = 100 * sqrt(mean(relErr(:,p).^2));
        end
        rows(iStrat).meanRmsePct = mean([rows(iStrat).rmsePct_Lsigma, ...
            rows(iStrat).rmsePct_Cps, rows(iStrat).rmsePct_Cg, rows(iStrat).rmsePct_Rac]);
    end
    Tmc = sortrows(struct2table(rows), 'meanRmsePct', 'ascend');
end

function thetaBest = estimate_theta(w, Hmeas, measStd, thetaTrue, GgFixed, channels)
    logTrue = log(thetaTrue);
    starts = [
         0,     0,     0,     0
         0.10, -0.10,  0.08,  0.08
        -0.12,  0.10, -0.08, -0.08
         0.22,  0.18, -0.16,  0.12
    ];
    bestCost = inf;
    thetaBest = thetaTrue;
    opts = optimset('Display','off', 'MaxIter', 700, 'MaxFunEvals', 2500, ...
        'TolX', 1e-8, 'TolFun', 1e-8);
    for i = 1:size(starts,1)
        x0 = logTrue + starts(i,:);
        obj = @(x) objective(x, w, Hmeas, measStd, thetaTrue, GgFixed, channels);
        [x, cost] = fminsearch(obj, x0, opts);
        if cost < bestCost
            bestCost = cost;
            thetaBest = exp(x);
        end
    end
end

function cost = objective(logTheta, w, Hmeas, measStd, thetaPrior, GgFixed, channels)
    theta = exp(logTheta);
    Hhat = select_channels(hft_twoport_y(w, theta, GgFixed), channels);
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

function Y = hft_twoport_y(w, theta, GgFixed)
    Lsigma = theta(1);
    Cps = theta(2);
    Cg = theta(3);
    Rac = theta(4);
    s = 1i*w(:).';
    ys = 1 ./ (Rac + s*Lsigma);
    yps = s * Cps;
    yg = GgFixed + s * Cg;
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

function S = multiport_sensitivity(w, theta, GgFixed, channels)
    H0 = select_channels(hft_twoport_y(w, theta, GgFixed), channels); %#ok<NASGU>
    nParam = numel(theta);
    S = zeros(numel(select_channels(hft_twoport_y(w, theta, GgFixed), channels)), nParam);
    step = 1e-4;
    for p = 1:nParam
        tp = theta; tm = theta;
        tp(p) = theta(p) * exp(step);
        tm(p) = theta(p) * exp(-step);
        Hp = select_channels(hft_twoport_y(w, tp, GgFixed), channels);
        Hm = select_channels(hft_twoport_y(w, tm, GgFixed), channels);
        S(:,p) = reshape((Hp - Hm) / (2*step), [], 1);
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

function plot_results(Tmc, fCand, idxPrior, idxNominal, idxUniform, dataDir)
    figure('Color','w','Position',[100,100,980,420]);
    bar(1:height(Tmc), Tmc.meanRmsePct);
    set(gca, 'XTick', 1:height(Tmc), 'XTickLabel', cellstr(Tmc.strategy));
    xtickangle(20);
    ylabel('Mean parameter RMSE (%)');
    title('Reduced 4-parameter MC verification after fixing Gg');
    grid on;
    saveas(gcf, fullfile(dataDir, 'oed_frequency_reduced_mc_rmse.png'));

    figure('Color','w','Position',[100,100,980,420]);
    semilogx(fCand(idxUniform), ones(size(idxUniform)), 'o'); hold on;
    semilogx(fCand(idxNominal), 1.08*ones(size(idxNominal)), 'x');
    semilogx(fCand(idxPrior), 1.16*ones(size(idxPrior)), '+');
    ylim([0.94, 1.22]);
    set(gca, 'YTick', [1, 1.08, 1.16], 'YTickLabel', {'uniform','nominal','prior'});
    grid on;
    xlabel('Frequency (Hz)');
    title('Selected frequencies for reduced 4-parameter experiment');
    saveas(gcf, fullfile(dataDir, 'oed_frequency_reduced_selected_points.png'));
end

function write_report(Tmc, dataDir)
    fid = fopen(fullfile(dataDir, 'oed_frequency_reduced_report.txt'), 'w');
    fprintf(fid, 'Reduced 4-parameter OED demo after fixing Gg\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, 'Monte Carlo ranking:\n');
    for i = 1:height(Tmc)
        fprintf(fid, '%s, K=%d, meanRmsePct=%.6e\n', ...
            Tmc.strategy(i), Tmc.K(i), Tmc.meanRmsePct(i));
    end
    fclose(fid);
end
