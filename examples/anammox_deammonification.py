"""Anaerobic deammonification with the asm3_2step_anammox model.

Anammox (anaerobic ammonium-oxidising) bacteria remove nitrogen autotrophically:
ammonium is oxidised with nitrite as the electron acceptor straight to dinitrogen
gas, with no organic carbon and a little nitrate produced for cell synthesis
(the canonical Strous et al. 1998 stoichiometry NH4 : NO2 : NO3 ~ 1 : 1.32 :
0.26). This is the second, anammox half of a partial-nitritation/anammox (PN/A)
sidestream process: an upstream aerobic step (the AOB nitritation already in the
model) converts part of the ammonium to nitrite, then anammox finishes it here.

Run a warm (30 C) anaerobic batch with ammonium and nitrite present and watch the
two substrates disappear into N2. Nitrite is the limiting acceptor (1.32 per
ammonium), so it is exhausted first.
"""
import jax.numpy as jnp
import numpy as np

import aquakin

net = aquakin.load_model("asm3_2step_anammox")
conditions = aquakin.SpatialConditions.uniform(T=303.15)  # 30 C sidestream
reactor = aquakin.BatchReactor(net, conditions)

# A nitritated sidestream stream: ammonium + nitrite (roughly the Strous ratio),
# no oxygen, no organic carbon, a healthy anammox population.
C0 = net.concentrations({
    "SO2": 0.0, "SNH4": 35.0, "SNO2": 40.0, "SNO3": 0.0, "SN2": 0.0,
    "XAMX": 250.0, "XAOB": 0.0, "XNOB": 0.0, "XH": 0.0, "XSTO": 0.0,
    "SS": 0.0, "SALK": 0.06,
})

t = jnp.linspace(0.0, 2.0, 9)
sol = reactor.solve(C0, params=net.default_parameters(), t_span=(0.0, 2.0), t_eval=t)

nh4 = np.asarray(sol.C_named("SNH4"))
no2 = np.asarray(sol.C_named("SNO2"))
no3 = np.asarray(sol.C_named("SNO3"))
n2 = np.asarray(sol.C_named("SN2"))

print("Anaerobic deammonification (anammox), 30 C")
print(f"{'t (d)':>6} {'SNH4':>7} {'SNO2':>7} {'SNO3':>7} {'SN2':>7}  (g N / m3)")
for i in range(len(t)):
    print(f"{float(t[i]):6.2f} {nh4[i]:7.1f} {no2[i]:7.1f} {no3[i]:7.1f} {n2[i]:7.1f}")

tin0 = nh4[0] + no2[0] + no3[0]
tin1 = nh4[-1] + no2[-1] + no3[-1]
print(f"\nInorganic N (NH4+NO2+NO3): {tin0:.1f} -> {tin1:.1f} g N/m3 "
      f"({100 * (tin0 - tin1) / tin0:.0f}% removed to N2, no organic carbon)")
print(f"Nitrate produced: {no3[-1]:.1f} g N/m3 "
      f"({100 * no3[-1] / (nh4[0] - nh4[-1]):.0f}% of the ammonium consumed "
      f"-- the Strous ~26% autotrophic-growth byproduct)")
