% Vector fitting with noisy HFT response and PWM-spectrum weights.
% Goal: compare uniform fitting against fitting weights implied by
% informative and weak PWM excitation conditions.

clear; clc; close all;
rng(20260707);

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
dataFolderName = char([25968 25454]); % Chinese folder name: data
dataDir = fullfile(projectDir, dataFolderName);
vfDir = fullfile(projectDir, char([22797 29616 39033 30446]), ...
    'SINTEF_Vector_Fitting_vfit3');

if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end
addpath(vfDir);

Ns = 600;
f = logspace(3, 7, Ns);
s = 1i * 2*pi*f;

theta.Lsigma = 2e-6;
theta.Cp = 200e-12;
theta.Rac = 0.2;

Ztrue = simple_impedance_response(2*pi*f, theta);

% Simple synthetic measurement noise: amplitude noise plus phase noise.
ampNoiseStd = 0.01;          % 1 percent magnitude noise
phaseNoiseStdDeg = 0.2;      % 0.2 degree phase noise
ampFactor = 1 + ampNoiseStd * randn(size(Ztrue));
phaseFactor = exp(1i * deg2rad(phaseNoiseStdDeg) * randn(size(Ztrue)));
Zmeas = Ztrue .* ampFactor .* phaseFactor;

cases(1).name = 'uniform';
cases(1).fs = nan;
cases(1).duty = nan;
cases(1).tr = nan;
cases(1).weight = ones(size(f));

cases(2).name = 'pwm_best_tr50ns_fs500k_D03';
cases(2).fs = 500e3;
cases(2).duty = 0.3;
cases(2).tr = 50e-9;
cases(2).weight = pwm_spectrum_weight(f, cases(2).fs, cases(2).duty, cases(2).tr);

cases(3).name = 'pwm_weak_tr500ns_fs50k_D07';
cases(3).fs = 50e3;
cases(3).duty = 0.7;
cases(3).tr = 500e-9;
cases(3).weight = pwm_spectrum_weight(f, cases(3).fs, cases(3).duty, cases(3).tr);

order = 6;
basePoles = -2*pi*logspace(3, 7, order);
opts = make_vf_opts();

rows = [];
fits = zeros(numel(cases), Ns);
for iCase = 1:numel(cases)
    poles = basePoles;

    % Combine response scaling with the chosen spectral emphasis.
    wFit = cases(iCase).weight;
    wFit = 0.05 + 0.95 * wFit ./ max(wFit);
    wFit = wFit ./ max(abs(Zmeas), 1e-9);

    rmserrHistory = nan(1, 5);
    fit = [];
    for iter = 1:numel(rmserrHistory)
        [SER, poles, rmserr, fit] = vectfit3(Zmeas, s, poles, wFit, opts); %#ok<ASGLU>
        rmserrHistory(iter) = rmserr;
    end
    fits(iCase,:) = fit;

    relToTrue = abs((fit - Ztrue) ./ max(abs(Ztrue), 1e-9));
    relToMeas = abs((fit - Zmeas) ./ max(abs(Zmeas), 1e-9));
    hiMask = f >= 1e6;
    pwmMask = cases(iCase).weight >= 0.1 * max(cases(iCase).weight);
    if iCase == 1
        pwmMask = true(size(f));
    end

    rows(iCase).caseName = string(cases(iCase).name); %#ok<SAGROW>
    rows(iCase).fitOrder = order;
    rows(iCase).meanRelErrTrue = mean(relToTrue);
    rows(iCase).maxRelErrTrue = max(relToTrue);
    rows(iCase).meanRelErrMeas = mean(relToMeas);
    rows(iCase).maxRelErrMeas = max(relToMeas);
    rows(iCase).meanRelErrTrueHighFreq = mean(relToTrue(hiMask));
    rows(iCase).meanRelErrTrueWeightedBand = mean(relToTrue(pwmMask));
    rows(iCase).finalRmse = rmserrHistory(end);
end

T = struct2table(rows);
writetable(T, fullfile(dataDir, 'vf_noise_weight_metrics.csv'));
save(fullfile(dataDir, 'vf_noise_weight_demo.mat'), 'T', 'cases', ...
    'f', 'Ztrue', 'Zmeas', 'fits', 'theta');

write_summary(T, dataDir);
plot_noise_weight_result(f, Ztrue, Zmeas, fits, cases, dataDir);

disp(T);

function opts = make_vf_opts()
    opts.relax = 1;
    opts.stable = 1;
    opts.asymp = 2;
    opts.skip_pole = 0;
    opts.skip_res = 0;
    opts.cmplx_ss = 1;
    opts.spy1 = 0;
    opts.spy2 = 0;
    opts.logx = 1;
    opts.logy = 1;
    opts.errplot = 0;
    opts.phaseplot = 0;
    opts.legend = 0;
end

function Z = simple_impedance_response(w, theta)
    Zs = theta.Rac + 1i*w*theta.Lsigma;
    Yp = 1i*w*theta.Cp;
    Z = 1 ./ (1./Zs + Yp);
end

function U2 = pwm_spectrum_weight(f, fs, duty, tr)
    x = f / fs;
    duty_envelope = abs(sin(pi*duty*x) ./ max(pi*x, 1e-12));
    rise_envelope = abs(sinc_local(f*tr));
    U2 = (duty_envelope .* rise_envelope).^2;
    U2 = U2(:).' / max(U2(:));
end

function y = sinc_local(x)
    y = ones(size(x));
    nz = abs(x) > 1e-12;
    y(nz) = sin(pi*x(nz)) ./ (pi*x(nz));
end

function write_summary(T, dataDir)
    fid = fopen(fullfile(dataDir, 'vf_noise_weight_summary.txt'), 'w');
    fprintf(fid, 'Noisy vector fitting with PWM weights\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    for k = 1:height(T)
        fprintf(fid, 'Case: %s\n', T.caseName(k));
        fprintf(fid, '  meanRelErrTrue = %.6e\n', T.meanRelErrTrue(k));
        fprintf(fid, '  maxRelErrTrue = %.6e\n', T.maxRelErrTrue(k));
        fprintf(fid, '  meanRelErrTrueHighFreq = %.6e\n', T.meanRelErrTrueHighFreq(k));
        fprintf(fid, '  meanRelErrTrueWeightedBand = %.6e\n', T.meanRelErrTrueWeightedBand(k));
        fprintf(fid, '  finalRmse = %.6e\n\n', T.finalRmse(k));
    end
    fclose(fid);
end

function plot_noise_weight_result(f, Ztrue, Zmeas, fits, cases, dataDir)
    colors = lines(numel(cases));

    figure('Color', 'w', 'Position', [100, 100, 1000, 650]);
    subplot(2,1,1);
    loglog(f, abs(Ztrue), 'k-', 'LineWidth', 1.8); hold on;
    loglog(f, abs(Zmeas), '.', 'Color', [0.65 0.65 0.65], 'MarkerSize', 4);
    for k = 1:numel(cases)
        loglog(f, abs(fits(k,:)), '--', 'Color', colors(k,:), 'LineWidth', 1.2);
    end
    grid on;
    xlabel('Frequency (Hz)');
    ylabel('|Z| (Ohm)');
    legendEntries = [{'True model', 'Noisy samples'}, {cases.name}];
    legend(legendEntries, 'Location', 'best');
    title('Noisy HFT response fitted with different weights');

    subplot(2,1,2);
    semilogx(f, angle(Ztrue)*180/pi, 'k-', 'LineWidth', 1.8); hold on;
    for k = 1:numel(cases)
        semilogx(f, angle(fits(k,:))*180/pi, '--', 'Color', colors(k,:), 'LineWidth', 1.2);
    end
    grid on;
    xlabel('Frequency (Hz)');
    ylabel('Phase (deg)');
    saveas(gcf, fullfile(dataDir, 'vf_noise_weight_fit.png'));

    figure('Color', 'w', 'Position', [100, 100, 950, 430]);
    for k = 1:numel(cases)
        relErr = abs((fits(k,:) - Ztrue) ./ max(abs(Ztrue), 1e-9));
        loglog(f, relErr, 'LineWidth', 1.3, 'Color', colors(k,:)); hold on;
    end
    grid on;
    xlabel('Frequency (Hz)');
    ylabel('Relative error to true model');
    legend({cases.name}, 'Location', 'best');
    title('Effect of fitting weights under noisy samples');
    saveas(gcf, fullfile(dataDir, 'vf_noise_weight_error.png'));

    figure('Color', 'w', 'Position', [100, 100, 950, 430]);
    for k = 2:numel(cases)
        semilogx(f, cases(k).weight ./ max(cases(k).weight), 'LineWidth', 1.4); hold on;
    end
    grid on;
    xlabel('Frequency (Hz)');
    ylabel('Normalized PWM weight');
    legend({cases(2:end).name}, 'Location', 'best');
    title('PWM spectral weights used for fitting');
    saveas(gcf, fullfile(dataDir, 'vf_noise_weight_pwm_weights.png'));
end
