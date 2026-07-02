"""Core data structures and AST machinery (no Pydantic dependency)."""

from aquakin.core.conditions import SpatialConditions
from aquakin.core.context import CompileContext
from aquakin.core.model import CompiledModel, compile_model

__all__ = ["CompileContext", "CompiledModel", "SpatialConditions", "compile_model"]
