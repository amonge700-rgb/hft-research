% Vector fitting reproduction on a first-version HFT impedance response.
% This script uses SINTEF vectfit3.m and writes outputs without touching
% the previous PWM-sweep figures.

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
dataFolderName = char([25968 25454]); % Chinese folder name: data
dataDir = fullfile(projectDir, dataFolderName);
vfDir = fullfile(projectDir, char([22797 29616 39033 30446]), ...
    'SINTEF_Vector_Fitting_vfit3'); % Chinese folder name: reproduction project

if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end
addpath(vfDir);

% Frequency response to fit.
Ns = 600;
f = logspace(3, 7, Ns); % 1 kHz to 10 MHz
s = 1i * 2*pi*f;

theta.Lsigma = 2e-6;
theta.Cp = 200e-12;
theta.Rac = 0.2;

Z = simple_impedance_response(2*pi*f, theta);

% Initial poles for vector fitting.
order = 6;
poles = -2*pi*logspace(3, 7, order);
weight = 1 ./ max(abs(Z), 1e-9);

opts.relax = 1;
opts.stable = 1;
opts.asymp = 2;      % D term only. Impedance should not need E*s here.
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

fit = [];
rmserrHistory = nan(1, 5);
for iter = 1:numel(rmserrHistory)
    [SER, poles, rmserr, fit] = vectfit3(Z, s, poles, weight, opts);
    rmserrHistory(iter) = rmserr;
end

relErr = abs((fit - Z) ./ max(abs(Z), 1e-9));
maxRelErr = max(relErr);
meanRelErr = mean(relErr);

summaryPath = fullfile(dataDir, 'vf_hft_demo_summary.txt');
fid = fopen(summaryPath, 'w');
fprintf(fid, 'Vector fitting HFT demo\n');
fprintf(fid, 'Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Model: Z = (Rac + s*Lsigma) parallel Cp\n');
fprintf(fid, 'Lsigma = %.6e H\n', theta.Lsigma);
fprintf(fid, 'Cp = %.6e F\n', theta.Cp);
fprintf(fid, 'Rac = %.6e Ohm\n', theta.Rac);
fprintf(fid, 'Frequency range = %.3e to %.3e Hz\n', min(f), max(f));
fprintf(fid, 'Vector fitting order = %d\n', order);
fprintf(fid, 'RMS error history:\n');
fprintf(fid, '  %.6e\n', rmserrHistory);
fprintf(fid, 'Mean relative error = %.6e\n', meanRelErr);
fprintf(fid, 'Max relative error = %.6e\n', maxRelErr);
fprintf(fid, 'Final poles:\n');
for k = 1:numel(poles)
    fprintf(fid, '  %.9e %+ .9ei\n', real(poles(k)), imag(poles(k)));
end
fclose(fid);

tablePath = fullfile(dataDir, 'vf_hft_demo_response.csv');
T = table(f(:), real(Z(:)), imag(Z(:)), real(fit(:)), imag(fit(:)), relErr(:), ...
    'VariableNames', {'f_Hz','Z_real','Z_imag','fit_real','fit_imag','relative_error'});
writetable(T, tablePath);

save(fullfile(dataDir, 'vf_hft_demo_SER.mat'), 'SER', 'poles', ...
    'rmserrHistory', 'theta', 'f', 'Z', 'fit', 'relErr');

plot_fit_result(f, Z, fit, relErr, dataDir);

disp(['Generated ', summaryPath]);
disp(['Generated ', tablePath]);
disp(['Mean relative error: ', num2str(meanRelErr, '%.3e')]);
disp(['Max relative error: ', num2str(maxRelErr, '%.3e')]);

function Z = simple_impedance_response(w, theta)
    Zs = theta.Rac + 1i*w*theta.Lsigma;
    Yp = 1i*w*theta.Cp;
    Z = 1 ./ (1./Zs + Yp);
end

function plot_fit_result(f, Z, fit, relErr, dataDir)
    figure('Color', 'w', 'Position', [100, 100, 960, 640]);

    subplot(2,1,1);
    loglog(f, abs(Z), 'k-', 'LineWidth', 1.6); hold on;
    loglog(f, abs(fit), 'r--', 'LineWidth', 1.3);
    grid on;
    xlabel('Frequency (Hz)');
    ylabel('|Z| (Ohm)');
    legend('Original model', 'Vector fitting', 'Location', 'best');
    title('HFT impedance response fitted by vectfit3');

    subplot(2,1,2);
    semilogx(f, angle(Z)*180/pi, 'k-', 'LineWidth', 1.6); hold on;
    semilogx(f, angle(fit)*180/pi, 'r--', 'LineWidth', 1.3);
    grid on;
    xlabel('Frequency (Hz)');
    ylabel('Phase (deg)');

    saveas(gcf, fullfile(dataDir, 'vf_hft_demo_fit.png'));

    figure('Color', 'w', 'Position', [100, 100, 900, 420]);
    loglog(f, relErr, 'LineWidth', 1.4);
    grid on;
    xlabel('Frequency (Hz)');
    ylabel('Relative error');
    title('Vector fitting relative error');
    saveas(gcf, fullfile(dataDir, 'vf_hft_demo_error.png'));
end
