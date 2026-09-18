# EXP-022: model-mismatch robust turn-level inverse

This experiment keeps the EXP-021 COMSOL matrices as geometry-specific truth
but deliberately makes the forward plant richer than the inverse model.

Forward-only nuisance effects:

- temperature and square-root frequency-dependent copper resistance;
- asymmetric turn-to-ground stray capacitance;
- frequency-dependent voltage/current probe gain and delay;
- multi-window current noise.

The inverse decoder still contains only the five physically grouped EXP-021
coordinates.  Geometry-disjoint tests compare nominal, pure physics, direct
edge-centric GNN and GNN-regularized physics refinement.

