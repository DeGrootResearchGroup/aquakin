"""Studies: analyses of a flowsheet project saved in the project with their results.

So far this holds the result files (:mod:`aquakin.study.files`), each value
keyed by name.
"""

from aquakin.study.files import (
    read_matrix,
    read_named_values,
    read_trajectory,
    write_matrix,
    write_named_values,
    write_trajectory,
)

__all__ = [
    "read_matrix",
    "read_named_values",
    "read_trajectory",
    "write_matrix",
    "write_named_values",
    "write_trajectory",
]
