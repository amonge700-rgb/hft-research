% Online PWM identification on the Liu 2016 wideband mechanism model.
%
% Literature anchor:
%   C. Liu et al., IEEE TPEL, 2016, Tables II-IV.
%
% Experiment:
%   time-domain LV voltage/current under several binary excitations
%       -> averaged cross-spectrum estimate of Z_in(jw)
%       -> dual-band feature extraction (f1 and f2)
%       -> comparison with the paper targets and the model sweep truth.
%
% The model is a table-parameter mechanism-model reproduction. It is not
% raw measured data and it is not the authors' complete FEM/circuit netlist.

clear; clc; close all;
rng(20260713);

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
dataDir = fullfile(projectDir, char([25968 25454]));
noteDir = fullfile(projectDir, char([31508 35760]));
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
if ~exist(noteDir, 'dir'), mkdir(noteDir); end

paper.n = 91.3;
paper.Lm = 4.587e-3;
paper.Ls0 = 42.45e-3;
paper.Rm = 422.8;
paper.Zs2_R = 37.8;
C.C12 = 8.3e-12;   C.C34 = 13.9e-12;  C.C24 = 93.0e-12;
C.C13 = 95.5e-12;  C.C23 = 61.7e-12;  C.C14 = 59.1e-12;
C.C10 = 161.1e-12; C.C20 = 160.6e-12; C.C30 = 35.3e-12; C.C40 = 29.9e-12;
paper.C = C;

conditions = [
    struct('name',"no_winding_grounded",  'groundedNodes',[],   'paperF1',2.5e3,'paperF2',74e3)
    struct('name',"lv_winding_grounded",  'groundedNodes',2,    'paperF1',2.5e3,'paperF2',74e3)
    struct('name',"hv_winding_grounded",  'groundedNodes',4,    'paperF1',2.0e3,'paperF2',60e3)
    struct('name',"both_windings_grounded",'groundedNodes',[2 4],'paperF1',1.8e3,'paperF2',53e3)
];

sim.fs = 1e6;
sim.N = 2^17;                 % 131 ms long window; df = 7.63 Hz
sim.nAvg = 10;
sim.voltageNoiseRel = 5e-4;
sim.currentNoiseRel = 3e-3;
sim.energyThreshold = 1e-6;   % bins below this level are not identifiable
sim.fEval = logspace(log10(700), log10(150e3), 2600);
methods = ["single_steep_pulse", "conventional_20k_pwm", ...
    "random_prbs", "targeted_binary_pwm", "dual_window_targeted"];

rows = struct([]);
curves = struct([]);
row = 0;
curveIdx = 0;

for cidx = 1:numel(conditions)
    cond = conditions(cidx);
    Ztrue = input_impedance_lv(sim.fEval, paper, cond.groundedNodes);
    truth = extract_features(sim.fEval, Ztrue);

    for midx = 1:numel(methods)
        method = methods(midx);
        rec = reconstruct_impedance(method, cond, paper, sim);
        feat = extract_features(rec.f, rec.Z);

        row = row + 1;
        rows(row).condition = cond.name; %#ok<SAGROW>
        rows(row).method = method;
        rows(row).paper_f1_Hz = cond.paperF1;
        rows(row).paper_f2_Hz = cond.paperF2;
        rows(row).sweep_f1_Hz = truth.f1;
        rows(row).sweep_f2_Hz = truth.f2;
        rows(row).online_f1_Hz = feat.f1;
        rows(row).online_f2_Hz = feat.f2;
        rows(row).errPct_f1_vs_sweep = 100*(feat.f1-truth.f1)/truth.f1;
        rows(row).errPct_f2_vs_sweep = 100*(feat.f2-truth.f2)/truth.f2;
        rows(row).errPct_f1_vs_paper = 100*(feat.f1-cond.paperF1)/cond.paperF1;
        rows(row).errPct_f2_vs_paper = 100*(feat.f2-cond.paperF2)/cond.paperF2;
        rows(row).bandEnergy_f1 = rec.bandEnergy(1);
        rows(row).bandEnergy_f2 = rec.bandEnergy(2);
        rows(row).coverage_f1 = rec.coverage(1);
        rows(row).coverage_f2 = rec.coverage(2);

        curveIdx = curveIdx + 1;
        curves(curveIdx).condition = cond.name; %#ok<SAGROW>
        curves(curveIdx).method = method;
        curves(curveIdx).f = rec.f;
        curves(curveIdx).Z = rec.Z;
        curves(curveIdx).inputEnergy = rec.inputEnergy;
        curves(curveIdx).Ztrue = Ztrue;
        curves(curveIdx).truth = truth;
    end
end

T = struct2table(rows);
rankRows = struct([]);
for midx = 1:numel(methods)
    q = T(T.method == methods(midx),:);
    e1 = q.errPct_f1_vs_sweep; e2 = q.errPct_f2_vs_sweep;
    rankRows(midx).method = methods(midx); %#ok<SAGROW>
    rankRows(midx).GroupCount = height(q);
    rankRows(midx).meanAbsErr_f1 = mean(abs(e1),'omitnan');
    rankRows(midx).meanAbsErr_f2 = mean(abs(e2),'omitnan');
    rankRows(midx).missingRate = mean([isnan(e1);isnan(e2)]);
    % Missing bands receive a 100 percentage-point penalty. This prevents
    % an excitation with no usable spectrum from ranking artificially well.
    rankRows(midx).penalizedScore = mean(abs([e1;e2]),'omitnan') + ...
        100*rankRows(midx).missingRate;
end
methodSummary = sortrows(struct2table(rankRows),"penalizedScore");

writetable(T, fullfile(dataDir, 'wideband_pwm_online_identification_summary.csv'));
writetable(methodSummary, fullfile(dataDir, 'wideband_pwm_online_method_ranking.csv'));
save(fullfile(dataDir, 'wideband_pwm_online_identification.mat'), ...
    'paper','conditions','sim','methods','T','methodSummary','curves');

plot_reconstructed_curves(curves, conditions, methods, sim, dataDir);
plot_error_summary(T, methods, dataDir);
plot_excitation_energy(curves, methods, dataDir);
write_experiment_note(T, methodSummary, sim, noteDir);

disp(T);
disp(methodSummary);

function rec = reconstruct_impedance(method, cond, paper, sim)
    if method == "dual_window_targeted"
        low = sim;
        low.N = 2^18;
        low.fs = 250e3;
        low.fEval = logspace(log10(700), log10(12e3), 900);
        high = sim;
        high.N = 2^16;
        high.fs = 1e6;
        high.fEval = logspace(log10(12e3), log10(150e3), 1300);
        recLow = reconstruct_single_window("targeted_low_band", cond, paper, low);
        recHigh = reconstruct_single_window("targeted_high_band", cond, paper, high);
        rec.f = [recLow.f, recHigh.f];
        rec.Z = [recLow.Z, recHigh.Z];
        rec.inputEnergy = [recLow.inputEnergy, recHigh.inputEnergy];
        rec.bandEnergy = [recLow.bandEnergy(1), recHigh.bandEnergy(2)];
        rec.coverage = [recLow.coverage(1), recHigh.coverage(2)];
    else
        rec = reconstruct_single_window(method, cond, paper, sim);
    end
end

function rec = reconstruct_single_window(method, cond, paper, sim)
    fs = sim.fs;
    N = sim.N;
    nPos = floor(N/2)+1;
    fPos = (0:nPos-1)*fs/N;
    fSafe = fPos;
    fSafe(1) = max(fs/N/100, 1e-3);
    Zpos = input_impedance_lv(fSafe, paper, cond.groundedNodes);
    YinPos = 1 ./ Zpos;
    YinPos(1) = YinPos(2);

    Svv = zeros(1,nPos);
    Siv = zeros(1,nPos);
    for a = 1:sim.nAvg
        v = make_excitation(method, sim, a, cond);
        V = fft(v);
        Ifull = build_full_spectrum(YinPos, N).*V;
        i = real(ifft(Ifull));
        vm = v + sim.voltageNoiseRel*max(std(v),1e-12)*randn(size(v));
        im = i + sim.currentNoiseRel*max(std(i),1e-12)*randn(size(i));
        Vm = fft(vm); Im = fft(im);
        Vp = Vm(1:nPos); Ip = Im(1:nPos);
        Svv = Svv + abs(Vp).^2;
        Siv = Siv + Ip.*conj(Vp);
    end

    lambda = 1e-10*max(Svv);
    YinEst = Siv ./ (Svv + lambda);
    Zest = 1 ./ max_complex(YinEst, 1e-12);
    energy = Svv/max(Svv);
    valid = fPos >= min(sim.fEval) & fPos <= max(sim.fEval) & energy > 1e-10;
    if nnz(valid) < 20
        valid = fPos >= min(sim.fEval) & fPos <= max(sim.fEval);
    end
    rec.f = sim.fEval;
    rec.Z = interp_complex(fPos(valid), Zest(valid), sim.fEval);
    rec.inputEnergy = interp1(log10(fPos(valid)), energy(valid), ...
        log10(sim.fEval), 'linear', 0);
    rec.bandEnergy = [band_mean(rec.f,rec.inputEnergy,0.8e3,4e3), ...
        band_mean(rec.f,rec.inputEnergy,35e3,100e3)];
    rec.coverage = [band_coverage(rec.f,rec.inputEnergy,0.8e3,4e3,sim.energyThreshold), ...
        band_coverage(rec.f,rec.inputEnergy,35e3,100e3,sim.energyThreshold)];
    rec.Z(rec.inputEnergy < sim.energyThreshold) = NaN;
end

function v = make_excitation(method, sim, avgIdx, cond)
    fs = sim.fs; N = sim.N; t = (0:N-1)/fs;
    switch method
        case "single_steep_pulse"
            t0 = 0.015 + 0.002*(avgIdx-1);
            width = 12e-3;
            tr = 2/fs;
            v = 0.5*(tanh((t-t0)/tr)-tanh((t-t0-width)/tr));
        case "conventional_20k_pwm"
            fc = 20e3;
            duty = 0.42 + 0.01*mod(avgIdx,5);
            v = double(mod(t*fc + 0.071*avgIdx,1) < duty)*2-1;
        case "random_prbs"
            chip = 8;
            levels = 2*(rand(1,ceil(N/chip))>0.5)-1;
            v = repelem(levels,chip); v = v(1:N);
        case "targeted_binary_pwm"
            phase = 2*pi*rand(1,4);
            x = 0.9*sin(2*pi*2.2e3*t+phase(1)) + ...
                0.55*sin(2*pi*53e3*t+phase(2)) + ...
                0.55*sin(2*pi*60e3*t+phase(3)) + ...
                0.55*sin(2*pi*74e3*t+phase(4));
            x = x + 0.08*randn(size(x));
            v = 2*(x>=0)-1;
        case "targeted_low_band"
            phase = 2*pi*rand(1,3);
            x = sin(2*pi*1.8e3*t+phase(1)) + ...
                sin(2*pi*2.0e3*t+phase(2)) + sin(2*pi*2.5e3*t+phase(3));
            v = 2*(x+0.12*randn(size(x))>=0)-1;
        case "targeted_high_band"
            centers = [53e3,60e3,74e3];
            x = zeros(size(t));
            for k = 1:numel(centers)
                x = x + sin(2*pi*centers(k)*t + 2*pi*rand);
            end
            v = 2*(x+0.12*randn(size(x))>=0)-1;
        otherwise
            error('Unknown excitation %s', method);
    end
    v = v-mean(v);
    v = v/max(abs(v)+eps);
end

function Z = input_impedance_lv(f, paper, groundedNodes)
    Z = zeros(size(f));
    for k = 1:numel(f)
        Y = total_nodal_admittance(f(k),paper);
        I = zeros(4,1);
        if ismember(2,groundedNodes)
            I(1)=1;
        else
            I(1)=1; I(2)=-1;
        end
        [V,ok] = solve_grounded(Y,I,groundedNodes);
        if ok, Z(k)=V(1)-V(2); else, Z(k)=NaN; end
    end
end

function Y = total_nodal_admittance(f,paper)
    w=2*pi*f; s=1i*w;
    C=paper.C; Cmat=zeros(4,4);
    pairs = [1 2 C.C12; 3 4 C.C34; 2 4 C.C24; 1 3 C.C13; ...
        2 3 C.C23; 1 4 C.C14];
    for k=1:size(pairs,1), Cmat=add_pair(Cmat,pairs(k,1),pairs(k,2),pairs(k,3)); end
    Cmat(1,1)=Cmat(1,1)+C.C10; Cmat(2,2)=Cmat(2,2)+C.C20;
    Cmat(3,3)=Cmat(3,3)+C.C30; Cmat(4,4)=Cmat(4,4)+C.C40;
    yl=1/(paper.Zs2_R+s*paper.Ls0);
    ym=1/paper.Rm+1/(s*paper.Lm); n=paper.n;
    Yp=[n^2*yl+ym,-n*yl;-n*yl,yl];
    B=[1 -1 0 0;0 0 1 -1];
    Y=1i*w*Cmat+B'*Yp*B;
end

function M=add_pair(M,a,b,c)
    M(a,a)=M(a,a)+c; M(b,b)=M(b,b)+c;
    M(a,b)=M(a,b)-c; M(b,a)=M(b,a)-c;
end

function [V,ok]=solve_grounded(Y,I,groundedNodes)
    keep=setdiff(1:4,groundedNodes); V=zeros(4,1);
    ok=rcond(Y(keep,keep))>1e-14;
    if ok, V(keep)=Y(keep,keep)\I(keep); end
end

function feat=extract_features(f,Z)
    y=movmean(log10(abs(Z)+1e-300),9,'omitnan');
    feat.f1=extreme_in_band(f,y,0.8e3,4e3,"max");
    feat.f2=extreme_in_band(f,y,35e3,100e3,"min");
end

function fx=extreme_in_band(f,y,fmin,fmax,kind)
    idx=find(f>=fmin & f<=fmax & isfinite(y));
    if isempty(idx), fx=NaN; return; end
    if kind=="max", [~,k]=max(y(idx)); else, [~,k]=min(y(idx)); end
    fx=f(idx(k));
end

function Xfull=build_full_spectrum(Xpos,N)
    if rem(N,2)==0
        Xfull=[Xpos,conj(Xpos(end-1:-1:2))];
    else
        Xfull=[Xpos,conj(Xpos(end:-1:2))];
    end
end

function z=max_complex(z,minMag)
    small=abs(z)<minMag;
    z(small)=minMag*exp(1i*angle(z(small)+eps));
end

function yi=interp_complex(x,y,xi)
    yi=interp1(log10(x),real(y),log10(xi),'linear','extrap') + ...
        1i*interp1(log10(x),imag(y),log10(xi),'linear','extrap');
end

function v=band_mean(f,x,fmin,fmax)
    idx=f>=fmin & f<=fmax;
    v=mean(x(idx),'omitnan');
end

function v=band_coverage(f,x,fmin,fmax,threshold)
    idx=f>=fmin & f<=fmax;
    v=mean(x(idx)>=threshold);
end

function plot_reconstructed_curves(curves,conditions,methods,sim,dataDir)
    figure('Color','w','Position',[60,40,1200,820]);
    colors=lines(numel(methods));
    for c=1:numel(conditions)
        subplot(2,2,c); hold on;
        idx0=find([curves.condition]==conditions(c).name,1);
        semilogx(sim.fEval,20*log10(abs(curves(idx0).Ztrue)),'k','LineWidth',2);
        for m=1:numel(methods)
            idx=find([curves.condition]==conditions(c).name & [curves.method]==methods(m),1);
            semilogx(curves(idx).f,20*log10(abs(curves(idx).Z)), ...
                'Color',colors(m,:),'LineWidth',0.9);
        end
        xline(conditions(c).paperF1,':k'); xline(conditions(c).paperF2,':k');
        grid on; xlim([700 150e3]); title(strrep(conditions(c).name,'_','\_'));
        xlabel('Frequency (Hz)'); ylabel('|Z_{in}| (dB-ohm)');
        if c==1, legend(['sweep truth',cellstr(methods)],'Location','best','FontSize',7); end
    end
    saveas(gcf,fullfile(dataDir,'wideband_pwm_online_reconstructed_curves.png'));
end

function plot_error_summary(T,methods,dataDir)
    E1=zeros(numel(methods),1); E2=E1;
    missing1=false(numel(methods),1); missing2=missing1;
    for m=1:numel(methods)
        q=T(T.method==methods(m),:);
        E1(m)=mean(abs(q.errPct_f1_vs_sweep));
        E2(m)=mean(abs(q.errPct_f2_vs_sweep));
        missing1(m)=any(isnan(q.errPct_f1_vs_sweep));
        missing2(m)=any(isnan(q.errPct_f2_vs_sweep));
    end
    figure('Color','w','Position',[90,90,920,440]);
    bar([E1,E2]); grid on; hold on;
    for m=1:numel(methods)
        if missing1(m), text(m-0.15,0.18,'N/A','Rotation',90,'HorizontalAlignment','center'); end
        if missing2(m), text(m+0.15,0.18,'N/A','Rotation',90,'HorizontalAlignment','center'); end
    end
    set(gca,'XTickLabel',cellstr(methods)); xtickangle(22);
    ylabel('Mean absolute frequency error (%)');
    legend('f1','f2','Location','best');
    saveas(gcf,fullfile(dataDir,'wideband_pwm_online_error_comparison.png'));
end

function plot_excitation_energy(curves,methods,dataDir)
    figure('Color','w','Position',[90,90,950,520]); hold on;
    colors=lines(numel(methods));
    for m=1:numel(methods)
        idx=find([curves.condition]=="no_winding_grounded" & [curves.method]==methods(m),1);
        semilogx(curves(idx).f,10*log10(curves(idx).inputEnergy+1e-14), ...
            'Color',colors(m,:),'LineWidth',1.1);
    end
    xline(2.5e3,':k','f1'); xline(74e3,':k','f2');
    grid on; xlim([700 150e3]); ylim([-120 5]);
    xlabel('Frequency (Hz)'); ylabel('Normalized input energy (dB)');
    legend(cellstr(methods),'Location','best');
    saveas(gcf,fullfile(dataDir,'wideband_pwm_online_excitation_energy.png'));
end

function write_experiment_note(T,ranking,sim,noteDir)
    path=fullfile(noteDir,'wideband_pwm_online_identification_record.md');
    fid=fopen(path,'w','n','UTF-8');
    fprintf(fid,'# Wideband mechanism model online PWM identification\n\n');
    fprintf(fid,'This experiment uses Liu 2016 Tables II-IV as the mechanism-model anchor.\n');
    fprintf(fid,'It reconstructs LV input impedance from time-domain voltage/current; no sweep data enter the online estimator.\n\n');
    fprintf(fid,'## Simulation settings\n\n- fs = %.0f Hz\n- N = %d\n- averages = %d\n',sim.fs,sim.N,sim.nAvg);
    fprintf(fid,'- voltage noise = %.4f relative RMS\n- current noise = %.4f relative RMS\n',sim.voltageNoiseRel,sim.currentNoiseRel);
    fprintf(fid,'- normalized excitation-energy threshold = %.1e\n\n',sim.energyThreshold);
    fprintf(fid,'## Method ranking\n\n');
    for k=1:height(ranking)
        fprintf(fid,'- %s: MAE f1 %.3f%%, f2 %.3f%%; missing %.1f%%; penalized score %.3f\n', ...
            ranking.method(k),ranking.meanAbsErr_f1(k),ranking.meanAbsErr_f2(k), ...
            100*ranking.missingRate(k),ranking.penalizedScore(k));
    end
    fprintf(fid,'\n## Interpretation\n\n');
    fprintf(fid,'The sweep solution is the model truth; Table IV is the literature target. ');
    fprintf(fid,'The comparison therefore separates online reconstruction error from the small table-model reproduction error.\n');
    fprintf(fid,'The dual-window method uses a long low-band record for f1 and a shorter high-band record for f2.\n');
    fprintf(fid,'This remains a simulation study and must later be validated with measured PWM voltage/current.\n');
    fclose(fid);
end
