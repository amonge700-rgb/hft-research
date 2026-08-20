% Literature-feature to online-pulse bridge demo.
%
% This script builds the first-layer innovation experiment:
%   literature-style offline broadband features
%       -> pulse-excited frequency-response reconstruction
%       -> feature-based parameter extraction.
%
% It does not claim to be a real transformer FEM model. The purpose is to
% connect literature observables such as Zoc, Zsc, and voltage transfer
% features to an online natural-excitation setting.

clear; clc; close all;
rng(20260712);

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
dataDir = fullfile(projectDir, char([25968 25454]));
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

paramNames = {'Lsigma','Cps','Cg1','Cg2','Rac'};

% Frequency grid used by the literature-style feature extraction stage.
nLog = 300;
fLog = logspace(3, 7, nLog);
wLog = 2*pi*fLog;

% A terminal-equivalent two-winding HF transformer model.
thetaTrue = [2.0e-6, 85e-12, 95e-12, 150e-12, 0.12];
fixed.Lm = 420e-6;
fixed.Rm = 6.0e3;
fixed.Gg1 = 0.6e-6;
fixed.Gg2 = 1.3e-6;
thetaNom = [2.2e-6, 75e-12, 110e-12, 130e-12, 0.18];

Htrue = literature_observables(wLog, thetaTrue, fixed);

% Offline literature-style observations.
offlineRelNoise = 0.003;
Hoffline = add_complex_noise(Htrue, offlineRelNoise);

% Feature frequencies are selected around resonance/anti-resonance/transfer
% features. This imitates literature methods based on Zoc, Zsc, and voltage
% transfer characteristic frequencies.
Kfeature = 54;
idxFeatureTrue = select_feature_indices(fLog, Htrue, Kfeature);
idxFeatureOffline = select_feature_indices(fLog, Hoffline, Kfeature);

% Online natural-excitation reconstruction from a short steep pulse.
pulse.fs = 120e6;
pulse.N = 2^16;
pulse.t0 = 1.0e-6;
pulse.width = 7.0e-6;
pulse.tr = 50e-9;
pulse.inputNoiseRel = 0.001;
pulse.outputNoiseRel = 0.006;
pulse.nAverages = 24;

[HpulseLog, pulseDiag] = reconstruct_observables_from_pulse(thetaTrue, fixed, fLog, pulse);
idxFeaturePulse = select_feature_indices(fLog, HpulseLog, Kfeature);
idxFeaturePulseReliable = select_reliable_feature_indices(fLog, HpulseLog, Kfeature, ...
    pulseDiag.validLogIdx, pulseDiag.relNoiseLog);

% Parameter extraction methods.
methods = struct([]);
methods(1).name = "offline_full_sweep";
methods(1).idx = 1:nLog;
methods(1).Hobs = Hoffline;
methods(1).relNoise = offlineRelNoise * ones(1, nLog);

methods(2).name = "offline_feature_points";
methods(2).idx = idxFeatureOffline;
methods(2).Hobs = Hoffline;
methods(2).relNoise = 0.006 * ones(1, nLog);

methods(3).name = "pulse_fft_full_valid";
methods(3).idx = pulseDiag.validLogIdx;
methods(3).Hobs = HpulseLog;
methods(3).relNoise = pulseDiag.relNoiseLog;

methods(4).name = "pulse_fft_feature_points";
methods(4).idx = idxFeaturePulse;
methods(4).Hobs = HpulseLog;
methods(4).relNoise = pulseDiag.relNoiseLog;

methods(5).name = "pulse_fft_reliable_features";
methods(5).idx = idxFeaturePulseReliable;
methods(5).Hobs = HpulseLog;
methods(5).relNoise = pulseDiag.relNoiseLog;

rows = [];
HfitByMethod = cell(numel(methods),1);
for m = 1:numel(methods)
    idx = methods(m).idx;
    Hobs = methods(m).Hobs(:,idx);
    relNoise = methods(m).relNoise(idx);
    measStd = relNoise(:).' .* max(abs(Hobs), 1e-12);
    [thetaHat, cost] = estimate_theta_features(wLog(idx), Hobs, measStd, thetaNom, fixed);
    Hfit = literature_observables(wLog, thetaHat, fixed);
    HfitByMethod{m} = Hfit;
    responseErr = mean(abs(Hfit(:)-Htrue(:)) ./ max(abs(Htrue(:)), 1e-12));
    paramErr = 100 * (thetaHat - thetaTrue) ./ thetaTrue;

    rows(m).method = string(methods(m).name); %#ok<AGROW>
    rows(m).K = numel(idx);
    rows(m).cost = cost;
    rows(m).meanResponseErrPct = 100 * responseErr;
    for p = 1:numel(paramNames)
        rows(m).(['hat_', paramNames{p}]) = thetaHat(p);
        rows(m).(['errPct_', paramNames{p}]) = paramErr(p);
    end
end
T = struct2table(rows);
writetable(T, fullfile(dataDir, 'literature_feature_online_bridge_summary.csv'));

save(fullfile(dataDir, 'literature_feature_online_bridge_demo.mat'), ...
    'T', 'thetaTrue', 'thetaNom', 'fixed', 'fLog', 'Htrue', 'Hoffline', ...
    'HpulseLog', 'idxFeatureTrue', 'idxFeatureOffline', 'idxFeaturePulse', ...
    'idxFeaturePulseReliable', 'pulse', 'pulseDiag', 'HfitByMethod', 'methods');

plot_bridge_results(fLog, Htrue, Hoffline, HpulseLog, idxFeatureOffline, idxFeaturePulseReliable, ...
    pulseDiag, T, paramNames, dataDir);
write_bridge_report(T, thetaTrue, thetaNom, fixed, pulse, dataDir);
disp(T);

function H = literature_observables(w, theta, fixed)
    Y = terminal_y_matrix(w, theta, fixed);
    Y11 = reshape(Y(1,1,:), 1, []);
    Y12 = reshape(Y(1,2,:), 1, []);
    Y21 = reshape(Y(2,1,:), 1, []);
    Y22 = reshape(Y(2,2,:), 1, []);
    Zsc = 1 ./ Y11;                         % secondary shorted, V2 = 0
    YinOc = Y11 - Y12 .* Y21 ./ Y22;        % secondary open, I2 = 0
    Zoc = 1 ./ YinOc;
    Hv = -Y21 ./ Y22;                       % open-circuit voltage transfer V2/V1
    H = [Zoc; Zsc; Hv];
end

function Y = terminal_y_matrix(w, theta, fixed)
    Lsigma = theta(1);
    Cps = theta(2);
    Cg1 = theta(3);
    Cg2 = theta(4);
    Rac = theta(5);
    s = 1i*w(:).';
    ys = 1 ./ (Rac + s*Lsigma);
    yps = s*Cps;
    yg1 = fixed.Gg1 + s*Cg1;
    yg2 = fixed.Gg2 + s*Cg2;
    ym = 1 ./ (fixed.Rm + s*fixed.Lm);
    Y11 = ym + yg1 + yps + ys;
    Y22 = yg2 + yps + ys;
    Y12 = -yps - ys;
    Y = zeros(2,2,numel(w));
    Y(1,1,:) = Y11;
    Y(2,2,:) = Y22;
    Y(1,2,:) = Y12;
    Y(2,1,:) = Y12;
end

function Hn = add_complex_noise(H, relNoise)
    sigma = relNoise .* max(abs(H), 1e-12);
    Hn = H + (randn(size(H)) + 1i*randn(size(H))) ./ sqrt(2) .* sigma;
end

function idx = select_feature_indices(f, H, K)
    x = log10(f(:).');
    score = zeros(size(x));
    for r = 1:size(H,1)
        y = log10(abs(H(r,:)) + 1e-30);
        dy = gradient(y, x);
        d2 = [0, abs(diff(sign(diff(y)))), 0];
        score = score + abs(dy) + 4*d2;
        for i = 2:(numel(y)-1)
            if (y(i) > y(i-1) && y(i) > y(i+1)) || (y(i) < y(i-1) && y(i) < y(i+1))
                score(i) = score(i) + 8;
            end
        end
    end

    % Keep coarse coverage even when feature scores cluster near one band.
    idx = unique(round(linspace(1, numel(f), min(16, K))));
    minSep = max(2, floor(numel(f)/(K*2.5)));
    [~, order] = sort(score, 'descend');
    for j = 1:numel(order)
        cand = order(j);
        if all(abs(cand - idx) >= minSep)
            idx(end+1) = cand; %#ok<AGROW>
        end
        if numel(idx) >= K
            break;
        end
    end
    if numel(idx) < K
        extra = setdiff(order, idx, 'stable');
        idx = [idx, extra(1:(K-numel(idx)))]; %#ok<AGROW>
    end
    idx = sort(idx(1:K));
end

function idx = select_reliable_feature_indices(f, H, K, validIdx, relNoise)
    idxCand = select_feature_indices(f, H, min(numel(f), 4*K));
    validMask = false(size(f));
    validMask(validIdx) = true;
    reliableMask = validMask & relNoise < 0.055;
    idx = idxCand(reliableMask(idxCand));

    % Keep logarithmic frequency coverage inside the actually excited band.
    coverage = find(reliableMask);
    if ~isempty(coverage)
        coarse = unique(round(linspace(min(coverage), max(coverage), min(18, K))));
        coarse = coarse(reliableMask(coarse));
        idx = unique([coarse(:).', idx(:).'], 'stable');
    end

    if numel(idx) < K
        score = 1 ./ max(relNoise, 1e-4);
        score(~reliableMask) = -inf;
        [~, order] = sort(score, 'descend');
        extra = setdiff(order, idx, 'stable');
        idx = [idx(:).', extra(1:min(K-numel(idx), numel(extra)))]; %#ok<AGROW>
    end

    if numel(idx) < K
        extra = setdiff(validIdx, idx, 'stable');
        idx = [idx(:).', extra(1:min(K-numel(idx), numel(extra)))]; %#ok<AGROW>
    end
    idx = sort(idx(1:min(K, numel(idx))));
end

function [HpulseLog, diag] = reconstruct_observables_from_pulse(theta, fixed, fLog, pulse)
    fs = pulse.fs;
    N = pulse.N;
    dt = 1/fs;
    t = (0:N-1)*dt;
    u = 0.5*(tanh((t-pulse.t0)/pulse.tr) - tanh((t-pulse.t0-pulse.width)/pulse.tr));
    u = u - mean(u);
    U = fft(u);
    nPos = floor(N/2)+1;
    fPos = (0:nPos-1)*(fs/N);
    validPos = fPos >= min(fLog) & fPos <= max(fLog) & abs(U(1:nPos)) > 0.01*max(abs(U(1:nPos)));
    HposTrue = literature_observables(2*pi*fPos, theta, fixed);
    HposEst = nan(size(HposTrue));
    yStore = zeros(size(HposTrue,1), N);
    for r = 1:size(HposTrue,1)
        Hfull = build_full_spectrum(HposTrue(r,:), N);
        y = real(ifft(Hfull .* U));
        Syu = zeros(1, nPos);
        Suu = zeros(1, nPos);
        for a = 1:pulse.nAverages
            yNoise = pulse.outputNoiseRel * max(std(y), 1e-12) * randn(size(y));
            uNoise = pulse.inputNoiseRel * max(std(u), 1e-12) * randn(size(u));
            Ymeas = fft(y + yNoise);
            Umeas = fft(u + uNoise);
            Upos = Umeas(1:nPos);
            Ypos = Ymeas(1:nPos);
            Syu = Syu + Ypos .* conj(Upos);
            Suu = Suu + abs(Upos).^2;
        end
        Htmp = Syu ./ max(Suu, 1e-20);
        Htmp(~validPos) = nan;
        HposEst(r,:) = Htmp;
        yStore(r,:) = y;
    end
    HpulseLog = zeros(size(HposTrue,1), numel(fLog));
    for r = 1:size(HposTrue,1)
        good = validPos & isfinite(real(HposEst(r,:))) & isfinite(imag(HposEst(r,:)));
        HpulseLog(r,:) = interp1(log10(fPos(good)), HposEst(r,good), log10(fLog), 'linear', 'extrap');
    end
    interpEnergy = interp1(log10(max(fPos(2:end), fPos(2))), abs(U(2:nPos)).^2, log10(fLog), 'linear', 'extrap');
    interpEnergy = interpEnergy / max(interpEnergy);
    relNoiseLog = min(0.004 ./ sqrt(pulse.nAverages * (0.02 + interpEnergy)), 0.12);
    validLogIdx = find(interpEnergy > 0.004);
    diag.t = t;
    diag.u = u;
    diag.yStore = yStore;
    diag.fPos = fPos;
    diag.validPos = validPos;
    diag.interpEnergy = interpEnergy;
    diag.relNoiseLog = relNoiseLog;
    diag.validLogIdx = validLogIdx;
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

function [thetaBest, bestCost] = estimate_theta_features(w, Hobs, measStd, thetaNom, fixed)
    logNom = log(thetaNom);
    starts = [
         0,     0,     0,     0,     0
         0.12, -0.10,  0.08, -0.08,  0.10
        -0.12,  0.12, -0.08,  0.08, -0.10
         0.25,  0.18, -0.16,  0.14,  0.20
        -0.25, -0.18,  0.16, -0.14, -0.20
    ];
    opts = optimset('Display','off', 'MaxIter', 1200, 'MaxFunEvals', 5000, ...
        'TolX', 1e-9, 'TolFun', 1e-9);
    bestCost = inf;
    thetaBest = thetaNom;
    for i = 1:size(starts,1)
        x0 = logNom + starts(i,:);
        obj = @(x) feature_objective(x, w, Hobs, measStd, thetaNom, fixed);
        [x, cost] = fminsearch(obj, x0, opts);
        if cost < bestCost
            bestCost = cost;
            thetaBest = exp(x);
        end
    end
end

function cost = feature_objective(logTheta, w, Hobs, measStd, thetaNom, fixed)
    theta = exp(logTheta);
    Hhat = literature_observables(w, theta, fixed);
    r = [(real(Hhat-Hobs) ./ measStd), (imag(Hhat-Hobs) ./ measStd)];
    cost = sum(huber_loss(r(:), 1.5));
    cost = cost + 0.02 * sum(((logTheta - log(thetaNom)) ./ 0.9).^2);
end

function rho = huber_loss(r, delta)
    a = abs(r);
    rho = 0.5 * r.^2;
    idx = a > delta;
    rho(idx) = delta * (a(idx) - 0.5*delta);
end

function plot_bridge_results(f, Htrue, Hoffline, Hpulse, idxOffline, idxPulse, pulseDiag, T, paramNames, dataDir)
    obsNames = {'Zoc','Zsc','Hv'};
    figure('Color','w','Position',[80,80,1050,720]);
    for r = 1:3
        subplot(3,1,r);
        loglog(f, abs(Htrue(r,:)), 'k-', 'LineWidth', 1.4); hold on;
        loglog(f, abs(Hpulse(r,:)), 'b--', 'LineWidth', 1.0);
        loglog(f(idxOffline), abs(Hoffline(r,idxOffline)), 'ro', 'MarkerSize', 4);
        loglog(f(idxPulse), abs(Hpulse(r,idxPulse)), 'g+', 'MarkerSize', 5);
        grid on;
        ylabel(['|', obsNames{r}, '|']);
        if r == 1
            title('Literature observables and pulse-reconstructed features');
            legend('true sweep','pulse FFT reconstruction','offline features','pulse features','Location','best');
        end
        if r == 3
            xlabel('Frequency (Hz)');
        end
    end
    saveas(gcf, fullfile(dataDir, 'literature_bridge_feature_response.png'));

    figure('Color','w','Position',[80,80,980,420]);
    semilogx(f, pulseDiag.interpEnergy, 'LineWidth', 1.4); grid on;
    xlabel('Frequency (Hz)');
    ylabel('Normalized pulse spectral energy');
    title('Natural pulse spectrum available for online reconstruction');
    saveas(gcf, fullfile(dataDir, 'literature_bridge_pulse_spectrum.png'));

    figure('Color','w','Position',[80,80,980,520]);
    tUs = pulseDiag.t * 1e6;
    plot(tUs, pulseDiag.u ./ max(abs(pulseDiag.u)), 'k-', 'LineWidth', 1.2); hold on;
    plot(tUs, pulseDiag.yStore(3,:) ./ max(abs(pulseDiag.yStore(3,:))), 'b--', 'LineWidth', 1.0);
    xlim([0, 15]);
    grid on;
    xlabel('Time (us)');
    ylabel('Normalized waveform');
    title('Example steep-pulse input and open-circuit voltage-transfer output');
    legend('input pulse','output via Hv','Location','best');
    saveas(gcf, fullfile(dataDir, 'literature_bridge_pulse_waveform.png'));

    figure('Color','w','Position',[80,80,1050,430]);
    bar(categorical(cellstr(T.method)), T.meanResponseErrPct);
    ylabel('Full-band response error (%)');
    title('Parameter extraction quality from offline and pulse-reconstructed data');
    grid on;
    saveas(gcf, fullfile(dataDir, 'literature_bridge_response_error.png'));

    errMat = zeros(height(T), numel(paramNames));
    for p = 1:numel(paramNames)
        errMat(:,p) = T.(['errPct_', paramNames{p}]);
    end
    figure('Color','w','Position',[80,80,1050,460]);
    bar(errMat);
    set(gca, 'XTick', 1:height(T), 'XTickLabel', cellstr(T.method));
    xtickangle(20);
    ylabel('Parameter error (%)');
    legend(paramNames, 'Location','best');
    title('Recovered parameter error');
    grid on;
    saveas(gcf, fullfile(dataDir, 'literature_bridge_parameter_error.png'));
end

function write_bridge_report(T, thetaTrue, thetaNom, fixed, pulse, dataDir)
    fid = fopen(fullfile(dataDir, 'literature_feature_online_bridge_report.md'), 'w');
    fprintf(fid, '# Literature-feature to online-pulse bridge demo\n\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, '## Purpose\n\n');
    fprintf(fid, 'Build a first-layer innovation baseline: move literature-style offline broadband observables (Zoc, Zsc, voltage transfer) toward online steep-pulse reconstruction and feature-based parameter extraction.\n\n');
    fprintf(fid, '## Model\n\n');
    fprintf(fid, 'Estimated theta=[Lsigma,Cps,Cg1,Cg2,Rac]. Fixed Lm=%.3e H, Rm=%.3e Ohm.\n\n', fixed.Lm, fixed.Rm);
    fprintf(fid, 'Truth theta: %.4e, %.4e, %.4e, %.4e, %.4e.\n\n', thetaTrue);
    fprintf(fid, 'Initial nominal theta: %.4e, %.4e, %.4e, %.4e, %.4e.\n\n', thetaNom);
    fprintf(fid, 'Pulse: fs=%.3e Hz, N=%d, tr=%.3e s, width=%.3e s.\n\n', pulse.fs, pulse.N, pulse.tr, pulse.width);
    fprintf(fid, '## Results\n\n');
    fprintf(fid, '| method | K | meanResponseErrPct | err Lsigma | err Cps | err Cg1 | err Cg2 | err Rac |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|\n');
    for i = 1:height(T)
        fprintf(fid, '| %s | %d | %.4g | %.4g | %.4g | %.4g | %.4g | %.4g |\n', ...
            T.method(i), T.K(i), T.meanResponseErrPct(i), T.errPct_Lsigma(i), T.errPct_Cps(i), ...
            T.errPct_Cg1(i), T.errPct_Cg2(i), T.errPct_Rac(i));
    end
    fprintf(fid, '\n## Interpretation\n\n');
    fprintf(fid, 'The offline full sweep is the upper baseline. Feature-point extraction tests whether literature-style characteristic frequencies carry enough information. Pulse-FFT methods test whether the same observables can be reconstructed from a finite steep-pulse waveform. This is the bridge from offline literature methods to online natural excitation.\n');
    fclose(fid);
end
