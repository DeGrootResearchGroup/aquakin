"""Single-stage partial-nitritation/anammox (PN/A) sidestream deammonification.

A continuously-fed, low-dissolved-oxygen CSTR treating a high-ammonium sidestream
(e.g. digester reject water) by autotrophic nitrogen removal -- no organic carbon.
Three groups coexist in one tank: ammonia oxidisers (AOB) partially nitritate the
ammonium to nitrite, anammox combines the remaining ammonium with that nitrite to
N2, and nitrite oxidisers (NOB) are out-competed for nitrite and wash out.

The process is controlled by three couplings that this run demonstrates:

* Long HRT retains the slow-growing anammox. In a plain CSTR SRT = HRT, and the
  anammox growth rate (~0.08/d at 20 C, ~0.21/d at 30 C) sets a minimum HRT below
  which anammox washes out and the reactor collapses to nitrification. Here HRT =
  30 d keeps the dilution rate below the anammox growth rate.
* Low aeration holds the dissolved oxygen near ~0.05 gO2/m3. Anammox is strongly
  (reversibly) oxygen-inhibited, so DO must stay low; it also limits nitratation.
* Limited feed alkalinity caps nitritation at roughly half the ammonium (AOB stop
  when the alkalinity is spent), supplying anammox with a ~1:1.3 NH4:NO2 mix.

NOB suppression here is competitive: anammox keeps nitrite low, starving the NOB,
which (with the low DO) wash out. The residual effluent nitrate is the anammox
autotrophic-growth byproduct (~0.26 mol per mol N removed), not NOB activity.
"""
import jax.numpy as jnp

import aquakin
from aquakin.plant import Aeration, CSTRUnit, InfluentSeries, Plant

net = aquakin.load_network("asm3_2step_anammox")

# One low-DO CSTR. V/Q = 30 d HRT (retains anammox); kLa = 4/d holds DO ~ 0.05.
V, Q = 3000.0, 100.0
tank = CSTRUnit(
    name="reactor", network=net, volume=V, input_port_names=["in"],
    conditions={"T": 303.15},                       # 30 C sidestream
    aeration=Aeration(kla=4.0, species="SO2"),       # low DO; SO2 is the O2 species
    output_port="out",
)
plant = Plant("pna_sidestream")
plant.add_unit(tank)

# High-ammonium reject water, alkalinity-limited for partial nitritation, no COD.
NH4_IN = 500.0
feed = InfluentSeries.constant(net, {"SNH4": NH4_IN, "SALK": 0.065}, Q=Q, base="zero")
plant.add_influent("feed", feed, to="reactor.in")

# Seed the three functional groups; high anammox so it wins the start-up race.
seed = net.concentrations(
    {"XAOB": 100.0, "XNOB": 10.0, "XAMX": 300.0, "SO2": 0.3, "SALK": 0.065},
    base="zero",
)
y0 = plant.initial_state(overrides={"reactor": seed})

# Anammox is slow -- integrate to a long pseudo-steady state.
t_eval = jnp.linspace(0.0, 400.0, 401)
sol = plant.solve(t_span=(0.0, 400.0), t_eval=t_eval, y0=y0, max_steps=400_000)
eff = plant.stream(sol, "reactor.out")
g = lambda s: float(eff.C_named(s)[-1])

tin_eff = g("SNH4") + g("SNO2") + g("SNO3")
removal = 100.0 * (NH4_IN - tin_eff) / NH4_IN

print("Single-stage PN/A sidestream deammonification (30 C, HRT 30 d)")
print(f"  feed: {NH4_IN:.0f} g NH4-N/m3, no organic carbon")
print(f"  effluent  NH4 {g('SNH4'):6.1f}  NO2 {g('SNO2'):6.1f}  NO3 {g('SNO3'):6.1f}  "
      f"N2 {g('SN2'):6.1f}  (g N/m3), DO {g('SO2'):.3f} gO2/m3")
print(f"  biomass   AOB {g('XAOB'):6.1f}  NOB {g('XNOB'):6.1f}  AMX {g('XAMX'):6.1f}  (g COD/m3)")
print(f"  total inorganic-N removal: {removal:.0f}%  (autotrophic, no carbon)")
print(f"  NOB {'washed out' if g('XNOB') < 1.0 else 'present'}; "
      f"effluent nitrate is the anammox byproduct (~26% of N removed)")
