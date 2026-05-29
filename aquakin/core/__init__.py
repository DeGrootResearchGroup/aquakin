"""Core data structures and AST machinery (no Pydantic dependency)."""

from aquakin.core.conditions import SpatialConditions
from aquakin.core.context import CompileContext
from aquakin.core.network import CompiledNetwork, compile_network

__all__ = ["CompileContext", "CompiledNetwork", "SpatialConditions", "compile_network"]
