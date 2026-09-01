"""Prints the model provenance used by EXP-015 without modifying older experiments."""
from pathlib import Path
for name in ("experiment10_segmented_ladder_identification_20260721","experiment13_simulink_four_terminal_hft_20260725","experiment14_simscape_dab_validation_20260726"):
    p=Path(__file__).resolve().parents[2]/name
    print(name, "exists=",p.exists(),"files=",sum(1 for x in p.rglob('*') if x.is_file()))
