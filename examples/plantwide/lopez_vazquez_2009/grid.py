"""All 36 conditions of Lopez-Vazquez et al. (2009) Fig. 4/6, classified as in
Fig. 6 (dominant if >= 60%, else the two largest coexist), against the paper."""
import json

from model import run_sbr

NAME = dict(PAO="PAO", CMP="Competi", ALP="Alpha")
# Fig. 6 as printed: (feed acetate fraction, T, pH) -> outcome.
PAPER = {
    (1.0, 30, 6.0): "Competi", (1.0, 30, 7.0): "Competi", (1.0, 30, 7.5): "Competi+PAO",
    (1.0, 20, 6.0): "Competi+PAO", (1.0, 20, 7.0): "Competi+PAO", (1.0, 20, 7.5): "PAO",
    (0.75, 30, 6.0): "PAO+Alpha", (0.75, 30, 7.0): "PAO", (0.75, 30, 7.5): "PAO",
    (0.5, 30, 6.0): "Alpha", (0.5, 30, 7.0): "Alpha+PAO", (0.5, 30, 7.5): "PAO",
    (0.0, 30, 6.0): "Alpha", (0.0, 30, 7.0): "Alpha", (0.0, 30, 7.5): "Alpha",
    (0.0, 20, 6.0): "Alpha", (0.0, 20, 7.0): "Alpha+PAO", (0.0, 20, 7.5): "PAO",
}
def paper(f, T, pH):
    return PAPER.get((f, T, pH), "PAO")

def classify(fr):
    top = sorted(fr.items(), key=lambda kv: -kv[1])
    if top[0][1] >= 0.6:
        return NAME[top[0][0]]
    return "+".join(sorted([NAME[top[0][0]], NAME[top[1][0]]]))

if __name__ == "__main__":
    rows, agree = [], 0
    for f in (1.0, 0.75, 0.5, 0.0):
        for T in (10, 20, 30):
            for pH in (6.0, 7.0, 7.5):
                fr, _ = run_sbr(T, pH, f)
                got = classify(fr)
                want = "+".join(sorted(paper(f, T, pH).split("+")))
                ok = got == want
                agree += ok
                rows.append(dict(f_hac=f, T=T, pH=pH, PAO=round(fr["PAO"], 2), CMP=round(fr["CMP"], 2),
                                 ALP=round(fr["ALP"], 2), got=got, paper=want, ok=ok))
                print(f"HAc {f:4.2f} {T:2d}C pH{pH}: PAO {fr['PAO']:.2f} CMP {fr['CMP']:.2f} ALP {fr['ALP']:.2f}  -> {got:12s} paper {want:12s} {'ok' if ok else 'DIFF'}", flush=True)
    print(f"agree {agree}/36")
    json.dump(rows, open("grid.json", "w"), indent=1)
