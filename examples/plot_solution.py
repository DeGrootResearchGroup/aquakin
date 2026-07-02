"""Plotting a solution with ``sol.plot()`` (optional matplotlib).

``sol.plot(species)`` returns a matplotlib ``Axes`` -- no manual ``C_named`` /
unit-casting / time-axis boilerplate. The x-axis is labelled with the model's
time unit (days here; seconds for the ozone/UV models), and a single-species
plot labels the y-axis with that species' units.

Requires the optional ``plot`` extra:  ``pip install aquakin[plot]``.
"""

import jax.numpy as jnp

import aquakin


def main() -> None:
    net = aquakin.load_model("asm1")
    reactor = aquakin.BatchReactor(net, aquakin.OperatingConditions(T=293.15))
    sol = reactor.solve(net.default_concentrations(),
                        t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 50))

    # One species: y-axis auto-labelled "SNH [g_N/m³]", x-axis "time [d]".
    ax = sol.plot("SNH")

    # Several species on a second axes (legended); pass ax= to overlay instead.
    sol.plot(["SS", "XS"])

    ax.figure.savefig("asm1_snh.png", dpi=120)
    print("Wrote asm1_snh.png. In a notebook/REPL the Axes is returned for "
          "further tweaking (sol.plot('SNH').set_title(...)).")


if __name__ == "__main__":
    main()
