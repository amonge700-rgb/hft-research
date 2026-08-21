function d = make_exp14_settings()
% Reproducible first SPS-DAB operating point.
d.fs = 20e3;
d.Ts = 1/d.fs;
d.phi_deg = 20;
d.Vdc_p = 400;
d.Vdc_s = 100; % approximately Vdc_p / turns ratio
d.transition_time = 1e-6; % finite bridge edge used by the baseline model
d.local_solver_step = 1e-7; % 100 ns local Backward-Euler step
% Fast deterministic regression point.  This is deliberately not yet a
% long electro-thermal settling study; it validates topology, switching,
% sign conventions and instantaneous power conservation.
d.stop_cycles = 2;
d.discard_cycles = 1;
d.max_step = d.Ts/50;
d.model_name = 'hft_segmented_dab_switching_8p8s';
end
