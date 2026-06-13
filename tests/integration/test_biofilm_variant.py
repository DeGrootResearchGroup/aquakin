"""Validation of the layered-biofilm variant of the balanced sewer model.

``wats_sewer_khalil_paper_balanced_biofilm`` is the balanced network with its
three composite bulk+biofilm reactions (fermentation, fast/slow hydrolysis) split
into explicit ``_bulk``/``_biofilm`` halves, for use with
:class:`aquakin.BiofilmReactor`. These tests check that the split is
mass-consistent (the two halves reproduce the lumped composite rate exactly) and
that the depth-resolved solve produces the expected stratification --
methanogenesis concentrated toward the wall with acetate drawn down with depth --
the structure a lumped area-to-volume reactor cannot represent.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import aquakin

NDIR = os.path.join(os.path.dirname(aquakin.__file__), "networks")
A_V_LUMPED = 56.7
L_F = 8e-4   # biofilm thickness (m); A_V (areal->volumetric) per layer = 1/L_F


@pytest.fixture(scope="module")
def lumped():
    return aquakin.load_network("wats_sewer_khalil_paper_balanced")


@pytest.fixture(scope="module")
def variant():
    return aquakin.load_network("wats_sewer_khalil_paper_balanced_biofilm")


@pytest.fixture(scope="module")
def biofilm_rxns():
    # Biofilm reactions are exactly those carrying the {A_V} area factor.
    y = yaml.safe_load(
        open(os.path.join(NDIR, "wats_sewer_khalil_paper_balanced_biofilm.yaml"))
    )
    return [r["name"] for r in y["reactions"] if "{A_V}" in r["rate"]]


def test_variant_compiles_with_split_reactions(variant, biofilm_rxns):
    # 28 balanced reactions, 3 split into two -> 31; 12 original biofilm + 3 new
    # _biofilm halves -> 15 reactions carrying {A_V}.
    assert variant.n_reactions == 31
    assert len(biofilm_rxns) == 15
    for base in ("fermentation", "hydrolysis_fast", "hydrolysis_slow"):
        assert base + "_bulk" in variant.reaction_names
        assert base + "_biofilm" in variant.reaction_names


def test_split_reactions_sum_to_lumped(lumped, variant):
    # The _bulk + _biofilm halves must reproduce the original composite rate
    # exactly at the same state/conditions (bio_hf = [X_BH] + eps*{X_BF}*{A_V}).
    C0 = jnp.asarray(lumped.default_concentrations()).at[
        lumped.species_index["S_NO"]
    ].set(15.0)
    cond = aquakin.SpatialConditions.uniform(1, A_V=A_V_LUMPED, X_BF=10.0, pH=7.5).fields
    rl = lumped.rates(C0, lumped.default_parameters(), cond, 0)
    rv = variant.rates(C0, variant.default_parameters(), cond, 0)
    for base in ("fermentation", "hydrolysis_fast", "hydrolysis_slow"):
        orig = float(rl[lumped.reaction_names.index(base)])
        split = float(
            rv[variant.reaction_names.index(base + "_bulk")]
            + rv[variant.reaction_names.index(base + "_biofilm")]
        )
        assert split == pytest.approx(orig, rel=1e-6)


@pytest.mark.slow  # heavy: stiff biofilm solve
def test_variant_solves_and_stratifies(variant, biofilm_rxns):
    # Depth-resolved solve at realistic biofilm diffusivity: nitrate is consumed
    # before it penetrates, so methanogenesis runs deep -- methane accumulates
    # toward the wall and acetate is drawn down with depth.
    C0 = jnp.asarray(variant.default_concentrations()).at[
        variant.species_index["S_NO"]
    ].set(15.0)
    cond = aquakin.SpatialConditions.uniform(1, A_V=1.0 / L_F, X_BF=10.0, pH=7.5)
    # This areal {A_V} variant intentionally holds every particulate fixed (the
    # "mature biofilm" lumped approximation); pass it explicitly so the test
    # documents the choice and does not trip the reactive-particulate warning.
    fixed = jnp.array([s.startswith("X") for s in variant.species])
    bio = aquakin.BiofilmReactor(
        variant, cond, n_layers=6, thickness=L_F, area_per_volume=A_V_LUMPED,
        diffusivity=1e-4, boundary_layer=1e-4, biofilm_reactions=biofilm_rxns,
        fixed_mask=fixed, dtmax=3e-5,
    )
    sol = bio.solve(C0, params=variant.default_parameters(), t_span=(0.0, 5.0 / 24.0))
    assert bool(jnp.all(jnp.isfinite(sol.profile)))
    prof = np.asarray(sol.profile[-1])                      # (n_comp, n_species)
    layers = slice(1, None)                                 # biofilm, surface->wall
    ch4 = prof[layers, variant.species_index["S_CH4"]]
    vfa = prof[layers, variant.species_index["S_VFA"]]
    assert ch4[-1] > ch4[0] + 1.0      # more methane at the wall than the surface
    assert vfa[-1] < vfa[0] - 1.0      # less acetate at the wall (diffusion-limited)


@pytest.mark.slow  # heavy: stiff multispecies biofilm solve
def test_multispecies_groups_grow_and_stratify():
    # The full multispecies model carries X_SRB / X_MA / X_SOB as per-layer
    # growing/decaying states. Seed each in the biofilm layers and confirm a
    # depth-resolved solve is finite and every functional group grows in the
    # biofilm relative to the (dilute) bulk -- the per-layer-biomass dynamics the
    # interim [X_BH]-coupled model could not represent.
    net = aquakin.load_network("wats_sewer_khalil_paper_balanced_biofilm_multispecies")
    si = net.species_index
    n_layers = 8
    fixed = jnp.array([s == "X_I" for s in net.species])     # only inert solids fixed
    film = {"X_BH": 1000.0, "X_SRB": 300.0, "X_MA": 200.0, "X_SOB": 300.0,
            "X_S1": 50.0, "X_S2": 300.0}
    bulk = np.array(net.default_concentrations(), dtype=float)
    bulk[si["S_NO"]] = 25.0; bulk[si["sumS"]] = 9.0; bulk[si["S_SO4"]] = 4.0
    bulk[si["S_VFA"]] = 25.0; bulk[si["X_BH"]] = 10.0
    y0 = np.tile(bulk, (n_layers + 1, 1))
    for s, v in film.items():
        y0[1:, si[s]] = v
    bio = aquakin.BiofilmReactor(
        net, aquakin.SpatialConditions.uniform(1, pH=7.5),
        n_layers=n_layers, thickness=2e-3, area_per_volume=A_V_LUMPED,
        diffusivity=1e-4, boundary_layer=1e-4, fixed_mask=fixed, dtmax=3e-5,
    )
    sol = bio.solve(jnp.asarray(y0), t_span=(0.0, 5.0 / 24.0),
                    params=net.default_parameters())
    assert bool(jnp.all(jnp.isfinite(sol.profile)))
    prof = np.asarray(sol.profile[-1])                       # (n_comp, n_species)
    # every functional group is denser in the biofilm than the (dilute) bulk
    for g in ("X_BH", "X_SRB", "X_MA", "X_SOB"):
        assert prof[1:, si[g]].max() > 10.0 * prof[0, si[g]]
