from pathlib import Path
import mph

root = Path(__file__).resolve().parent
source = root / "model" / "revisions" / "exp20_teacher_4p4_axisym_C_LM_fine.mph"
client = mph.start(cores=4, version="6.4")
model = client.load(source)
j = model.java
es = j.component("comp1").physics("es")
boundaries = [
    [9,10,11,21], [12,13,14,22], [15,16,17,23], [18,19,20,24],
    [25,26,27,37], [28,29,30,38], [31,32,33,39], [34,35,36,40],
]
for i in range(1, 9):
    tag = f"electricpotential_{i}"
    if tag in list(es.feature().tags()):
        es.feature().remove(tag)
    term = es.create(f"term{i}", "Terminal")
    term.selection().set(boundaries[i-1])
    term.set("TerminalType", "Voltage")
    term.set("V0", f"V{i}")
model.parameter("case_id", "1")
j.study("stdC2").run()
for expression in [f"es.Q0_{i}" for i in range(1,9)] + [f"es.term{i}.Q0" for i in range(1,9)]:
    try:
        print(expression, model.evaluate(expression, dataset="研究 2//解 1"))
    except Exception as error:
        print(expression, "ERROR", str(error).splitlines()[0])
model.save(root / "model" / "revisions" / "terminal_charge_api_test.mph")
