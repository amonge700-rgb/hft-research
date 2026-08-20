% 高频变压器可辨识性分析 - MATLAB 骨架
% 第一版目标：比较不同 PWM 激励下 theta = [Lsigma, Cp, Rac] 的信息量。

clear; clc;

% Frequency grid
f = logspace(3, 7, 600);      % 1 kHz to 10 MHz
w = 2*pi*f;

% Nominal parameters
theta.Lsigma = 2e-6;          % H
theta.Cp = 200e-12;           % F
theta.Rac = 0.2;              % Ohm

% Example PWM settings
cases = [
    struct('name',"tr_50ns_fs_100k",  'fs',100e3,'duty',0.5,'tr',50e-9)
    struct('name',"tr_100ns_fs_100k", 'fs',100e3,'duty',0.5,'tr',100e-9)
    struct('name',"tr_200ns_fs_100k", 'fs',100e3,'duty',0.5,'tr',200e-9)
];

for k = 1:numel(cases)
    c = cases(k);
    U2 = pwm_spectrum_weight(f, c.fs, c.duty, c.tr);
    S = normalized_sensitivity(w, theta);
    I = fisher_information(S, U2);

    eigvals = eig(I);
    fprintf('\nCase: %s\n', c.name);
    fprintf('rank(I): %d\n', rank(I));
    fprintf('lambda_min(I): %.3e\n', min(eigvals));
    fprintf('cond(I): %.3e\n', cond(I));
    fprintf('det(I): %.3e\n', det(I));
end

function H = simple_impedance_response(w, theta)
    % First-version response: input impedance of series Rac+Lsigma
    % in parallel with Cp. Replace this with ladder-network response later.
    Zs = theta.Rac + 1i*w*theta.Lsigma;
    Yp = 1i*w*theta.Cp;
    H = 1 ./ (1./Zs + Yp);
end

function S = normalized_sensitivity(w, theta)
    % Columns correspond to dH/dlog(theta_i) ~= theta_i*dH/dtheta_i.
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

    % Optional scale by response magnitude to avoid only large-magnitude
    % frequencies dominating. Start simple; revise once real data exists.
    S = S ./ max(abs(H0(:)), 1e-12);
end

function U2 = pwm_spectrum_weight(f, fs, duty, tr)
    % Simple envelope-style weight for finite-rise-time PWM.
    % This is not an exact PWM spectrum; it is a first-pass broadband weight.
    x = f / fs;
    duty_envelope = abs(sin(pi*duty*x) ./ max(pi*x, 1e-12));
    rise_envelope = abs(sinc(f*tr));
    U2 = (duty_envelope .* rise_envelope).^2;
    U2 = U2(:) / max(U2(:));
end

function I = fisher_information(S, U2)
    W = U2(:);
    I = S' * (S .* W);
    I = real(I);
    I = I + 1e-12 * eye(size(I)); % small regularization for diagnostics
end
