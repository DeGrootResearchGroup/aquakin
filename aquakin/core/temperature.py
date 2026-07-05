"""Canonical temperature-correction primitives.

The equilibrium / rate constants used across the library are tabulated at a
reference temperature and corrected to the operating temperature in one of two
forms. Both live here so the formula and the physical constants have a single
home -- the pH solver, the kinetic and equilibrium precipitation engines, and
the model rate-constant corrections all call these instead of re-deriving the
exponent inline.

* **van't Hoff** -- a thermodynamic equilibrium constant ``K`` measured at
  ``T_ref`` is corrected with an enthalpy of reaction ``dH`` (J/mol):

      K(T) = K(T_ref) * exp(dH * van_t_hoff_factor(T))

  or, in natural-log space, ``ln K(T) = ln K(T_ref) + dH * van_t_hoff_factor(T)``.

* **Arrhenius / theta** -- a rate constant carries a per-degree factor
  ``theta`` referenced at ``ref_T``:

      k(T) = k(ref_T) * theta**(T - ref_T) = k(ref_T) * arrhenius_factor(...)
"""

import jax.numpy as jnp

# Universal gas constant, J/(mol K).
R_GAS = 8.314462618

# Reference temperature for the tabulated pK / pKsp values (K), i.e. 25 degC.
T_REF_THERMO = 298.15

# Natural log of 10 (pK <-> ln K conversions).
LN10 = jnp.log(10.0)


def van_t_hoff_factor(T_kelvin, T_ref=T_REF_THERMO):
    """The van't Hoff exponent factor ``(1/T_ref - 1/T) / R``.

    Multiply by the reaction enthalpy ``dH`` (J/mol) and exponentiate to correct
    an equilibrium constant from ``T_ref`` to ``T_kelvin``::

        K(T) = K(T_ref) * exp(dH * van_t_hoff_factor(T))

    ``dH = 0`` gives a factor that leaves the constant unchanged.

    Parameters
    ----------
    T_kelvin : scalar
        Absolute temperature in kelvin.
    T_ref : float, optional
        Reference temperature in kelvin (default :data:`T_REF_THERMO`, 25 degC).
    """
    return (1.0 / T_ref - 1.0 / T_kelvin) / R_GAS


def arrhenius_factor(T, ref_T, ln_theta):
    """The per-degree rate correction ``theta**(T - ref_T) = exp(ln_theta*(T-ref_T))``.

    Unity at ``T == ref_T``, so a model whose conditions sit at the reference
    temperature is unaffected. ``T``/``ref_T``/``ln_theta`` may be arrays (the
    correction is applied element-wise across a vector of rate constants).

    Parameters
    ----------
    T : scalar or array
        Operating temperature (same units as ``ref_T``; a difference is used, so
        kelvin and celsius give the same result).
    ref_T : scalar or array
        Reference temperature at which the rate-constant value is defined.
    ln_theta : scalar or array
        Natural log of the per-degree factor ``theta``.
    """
    return jnp.exp(ln_theta * (T - ref_T))
