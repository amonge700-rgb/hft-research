% PWM sweep identifiability analysis for a first-version HFT model.
% Model parameters: theta = [Lsigma, Cp, Rac].
% Output: CSV table and figures under ../数据.

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
dataFolderName = char([25968 25454]); % Chinese folder name: data
dataDir = fullfile(fileparts(scriptDir), dataFolderName);
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

% Frequency grid
f = logspace(3, 7, 800);      % 1 kHz to 10 MHz
w = 2*pi*f;

% Nominal parameters
theta.Lsigma = 2e-6;          % H
theta.Cp = 200e-12;           % F
theta.Rac = 0.2;              % Ohm

trList = [50e-9, 100e-9, 200e-9, 500e-9];
fsList = [50e3, 100e3, 200e3, 500e3];
dutyList = [0.3, 0.5, 0.7];

S = normalized_sensitivity(w, theta);

rows = [];
rowIndex = 0;
for iTr = 1:numel(trList)
    for iFs = 1:numel(fsList)
        for iDuty = 1:numel(dutyList)
            c.tr = trList(iTr);
            c.fs = fsList(iFs);
            c.duty = dutyList(iDuty);

            U2 = pwm_spectrum_weight(f, c.fs, c.duty, c.tr);
            I = fisher_information(S, U2);
            eigvals = eig(I);
            covApprox = pinv(I);
            corrApprox = covariance_to_correlation(covApprox);

            rowIndex = rowIndex + 1;
            rows(rowIndex).tr_ns = c.tr * 1e9; %#ok<SAGROW>
            rows(rowIndex).fs_kHz = c.fs / 1e3;
            rows(rowIndex).duty = c.duty;
            rows(rowIndex).rankI = rank(I);
            rows(rowIndex).lambdaMin = min(eigvals);
            rows(rowIndex).condI = cond(I);
            rows(rowIndex).detI = det(I);
            rows(rowIndex).traceInvI = trace(covApprox);
            rows(rowIndex).var_Lsigma = covApprox(1,1);
            rows(rowIndex).var_Cp = covApprox(2,2);
            rows(rowIndex).var_Rac = covApprox(3,3);
            rows(rowIndex).corr_Lsigma_Cp = corrApprox(1,2);
            rows(rowIndex).corr_Lsigma_Rac = corrApprox(1,3);
            rows(rowIndex).corr_Cp_Rac = corrApprox(2,3);
        end
    end
end

T = struct2table(rows);
T = sortrows(T, {'condI', 'lambdaMin'}, {'ascend', 'descend'});

csvPath = fullfile(dataDir, 'pwm_sweep_identifiability_results.csv');
writetable(T, csvPath);

fprintf('Generated %s\n', csvPath);
disp(T(1:min(10, height(T)), :));

plot_metric_surface(T, 'condI', 'Condition number cond(I)', ...
    fullfile(dataDir, 'pwm_sweep_condI.png'));
plot_metric_surface(T, 'lambdaMin', 'Minimum eigenvalue lambda_min(I)', ...
    fullfile(dataDir, 'pwm_sweep_lambdaMin.png'));
plot_metric_surface(T, 'detI', 'Information volume det(I)', ...
    fullfile(dataDir, 'pwm_sweep_detI.png'));

% Also save sensitivity and spectrum reference plots for the best case.
best = T(1,:);
U2best = pwm_spectrum_weight(f, best.fs_kHz*1e3, best.duty, best.tr_ns*1e-9);
plot_reference_curves(f, S, U2best, best, dataDir);

function H = simple_impedance_response(w, theta)
    Zs = theta.Rac + 1i*w*theta.Lsigma;
    Yp = 1i*w*theta.Cp;
    H = 1 ./ (1./Zs + Yp);
end

function S = normalized_sensitivity(w, theta)
    names = {'Lsigma','Cp','Rac'};
    H0 = simple_impedance_response(w, theta);
    S = zeros(numel(w), numel(names));

    for i = 1:numel(names)
        name = names{i};
        tp = theta;
        tm = theta;
        step = 1e-4;
        tp.(name) = theta.(name) * exp(step);
        tm.(name) = theta.(name) * exp(-step);
        Hp = simple_impedance_response(w, tp);
        Hm = simple_impedance_response(w, tm);
        S(:, i) = ((Hp - Hm) / (2*step)).';
    end

    S = S ./ max(abs(H0(:)), 1e-12);
end

function U2 = pwm_spectrum_weight(f, fs, duty, tr)
    x = f / fs;
    duty_envelope = abs(sin(pi*duty*x) ./ max(pi*x, 1e-12));
    rise_envelope = abs(sinc_local(f*tr));
    U2 = (duty_envelope .* rise_envelope).^2;
    U2 = U2(:) / max(U2(:));
end

function y = sinc_local(x)
    y = ones(size(x));
    nz = abs(x) > 1e-12;
    y(nz) = sin(pi*x(nz)) ./ (pi*x(nz));
end

function I = fisher_information(S, U2)
    W = U2(:);
    I = S' * (S .* W);
    I = real(I);
    I = I + 1e-12 * eye(size(I));
end

function C = covariance_to_correlation(Cov)
    d = sqrt(max(diag(Cov), eps));
    C = Cov ./ (d*d');
    C(~isfinite(C)) = 0;
end

function plot_metric_surface(T, metricName, plotTitle, outputPath)
    trVals = unique(T.tr_ns);
    fsVals = unique(T.fs_kHz);
    dutyVals = unique(T.duty);

    figure('Color', 'w', 'Position', [100, 100, 1050, 330]);
    for iDuty = 1:numel(dutyVals)
        duty = dutyVals(iDuty);
        Z = nan(numel(trVals), numel(fsVals));
        for iTr = 1:numel(trVals)
            for iFs = 1:numel(fsVals)
                idx = T.tr_ns == trVals(iTr) & T.fs_kHz == fsVals(iFs) & abs(T.duty-duty) < 1e-12;
                Z(iTr, iFs) = T.(metricName)(idx);
            end
        end
        subplot(1, numel(dutyVals), iDuty);
        imagesc(fsVals, trVals, log10(abs(Z)));
        set(gca, 'YDir', 'normal');
        xlabel('fs (kHz)');
        ylabel('tr (ns)');
        title(sprintf('D = %.1f', duty));
        colorbar;
    end
    sgtitle([plotTitle, ' (log10 scale)']);
    saveas(gcf, outputPath);
end

function plot_reference_curves(f, S, U2, best, dataDir)
    figure('Color', 'w', 'Position', [100, 100, 900, 420]);
    semilogx(f, abs(S(:,1)), 'LineWidth', 1.4); hold on;
    semilogx(f, abs(S(:,2)), 'LineWidth', 1.4);
    semilogx(f, abs(S(:,3)), 'LineWidth', 1.4);
    grid on;
    xlabel('Frequency (Hz)');
    ylabel('|normalized sensitivity|');
    legend('Lsigma','Cp','Rac', 'Location', 'best');
    title('Parameter sensitivity versus frequency');
    saveas(gcf, fullfile(dataDir, 'parameter_sensitivity.png'));

    figure('Color', 'w', 'Position', [100, 100, 900, 420]);
    semilogx(f, U2, 'LineWidth', 1.4);
    grid on;
    xlabel('Frequency (Hz)');
    ylabel('|U_{PWM}(f)|^2 normalized');
    title(sprintf('Best-case PWM spectrum weight: tr=%.0f ns, fs=%.0f kHz, D=%.1f', ...
        best.tr_ns, best.fs_kHz, best.duty));
    saveas(gcf, fullfile(dataDir, 'best_case_pwm_spectrum_weight.png'));
end
