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


def test_variant_solves_and_stratifies(variant, biofilm_rxns):
    # Depth-resolved solve at realistic biofilm diffusivity: nitrate is consumed
    # before it penetrates, so methanogenesis runs deep -- methane accumulates
    # toward the wall and acetate is drawn down with depth.
    C0 = jnp.asarray(variant.default_concentrations()).at[
        variant.species_index["S_NO"]
    ].set(15.0)
    cond = aquakin.SpatialConditions.uniform(1, A_V=1.0 / L_F, X_BF=10.0, pH=7.5)
    bio = aquakin.BiofilmReactor(
        variant, cond, n_layers=6, thickness=L_F, area_per_volume=A_V_LUMPED,
        diffusivity=1e-4, boundary_layer=1e-4, biofilm_reactions=biofilm_rxns,
        dtmax=3e-5,
    )
    sol = bio.solve(C0, variant.default_parameters(), t_span=(0.0, 5.0 / 24.0))
    assert bool(jnp.all(jnp.isfinite(sol.profile)))
    prof = np.asarray(sol.profile[-1])                      # (n_comp, n_species)
    layers = slice(1, None)                                 # biofilm, surface->wall
    ch4 = prof[layers, variant.species_index["S_CH4"]]
    vfa = prof[layers, variant.species_index["S_VFA"]]
    assert ch4[-1] > ch4[0] + 1.0      # more methane at the wall than the surface
    assert vfa[-1] < vfa[0] - 1.0      # less acetate at the wall (diffusion-limited)
