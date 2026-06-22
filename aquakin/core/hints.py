"""Close-match suggestion for unknown-name errors.

A single home for the "Did you mean: ...?" hint appended to ``KeyError`` /
``ValueError`` messages when a species / parameter / unit / stream name is not
found. Callers keep their own message prefix and exception type; this only
formats the suffix, so the hint wording (and its similarity threshold) is tuned
in one place.
"""

import difflib


def did_you_mean(name, choices, n: int = 3) -> str:
    """Return a ``" Did you mean: a, b?"`` suffix, or ``""`` if nothing is close.

    Parameters
    ----------
    name : str
        The unknown name the user supplied.
    choices : iterable of str
        The valid names to match against.
    n : int, optional
        Maximum number of suggestions (default 3).

    Returns
    -------
    str
        A leading-space suffix ready to append to an error message, empty when
        no close match is found.
    """
    matches = difflib.get_close_matches(name, list(choices), n=n)
    return f" Did you mean: {', '.join(matches)}?" if matches else ""
