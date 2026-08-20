% Recreate Monte Carlo parameter-identifiability figures from saved results.

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
dataDir = fullfile(projectDir, char([25968 25454]));

load(fullfile(dataDir, 'mc_parameter_identifiability.mat'), ...
    'summary', 'allThetaHat', 'thetaTrue', 'cases');

plot_mc_results(summary, allThetaHat, thetaTrue, cases, dataDir);
disp('Generated Monte Carlo figures from saved results.');

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
    saveas(gcf, fullfile(dataDir, 'mc_parameter_error_scatter.png'));

    figure('Color', 'w', 'Position', [100, 100, 980, 420]);
    yyaxis left;
    semilogy(1:height(summary), summary.lambdaMinPriorAvg, 'o-', 'LineWidth', 1.4);
    ylabel('Prior-averaged lambda min');
    yyaxis right;
    plot(1:height(summary), summary.meanRmsePct, 's-', 'LineWidth', 1.4);
    ylabel('Mean parameter RMSE (%)');
    set(gca, 'XTick', 1:height(summary), 'XTickLabel', cellstr(caseNames));
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
