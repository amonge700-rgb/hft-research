from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from hft_graph.reference_models import make_reference_graph
from hft_graph.visualization import save_graph_json, save_graph_svg
(ROOT/"results").mkdir(exist_ok=True)
for n in (4,8):
    g=make_reference_graph(n); save_graph_json(g,ROOT/"results"/f"graph_{n}x{n}.json"); save_graph_svg(g,ROOT/"results"/f"graph_{n}x{n}.svg")
