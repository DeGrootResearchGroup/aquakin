"""Fast unit tests for DGSM progress logging.

The ``progress=`` sampling loops in ``plant.steady_state_dgsm`` / ``dynamic_dgsm``
route their progress through :mod:`logging` (not ``print``) so a caller can
silence or redirect it. The cadence gate and the emitted record are exercised
here directly on ``_log_dgsm_progress`` -- no plant solve.
"""

import logging

import aquakin  # noqa: F401  (ensures the package logging setup runs)
from aquakin.plant.sensitivity import _log_dgsm_progress

_LOGGER = "aquakin.plant.sensitivity"


def test_emits_one_info_record_at_the_cadence(caplog):
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _log_dgsm_progress("steady_state_dgsm", 5, 10, 100, " (last: ptc)")
    records = [r for r in caplog.records if r.name == _LOGGER]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO  # below the default WARNING -> opt-in
    assert "steady_state_dgsm" in caplog.text
    assert "10/100 samples" in caplog.text
    assert "last: ptc" in caplog.text


def test_silent_off_cadence(caplog):
    # done (7) is not a multiple of progress (5) -> no record.
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _log_dgsm_progress("dynamic_dgsm", 5, 7, 100)
    assert [r for r in caplog.records if r.name == _LOGGER] == []


def test_silent_when_progress_is_none(caplog):
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _log_dgsm_progress("dynamic_dgsm", None, 10, 100)
    assert [r for r in caplog.records if r.name == _LOGGER] == []


def test_extra_is_optional():
    # The dynamic screen passes no `extra`; the message still formats cleanly.
    caplog_logger = logging.getLogger(_LOGGER)
    caplog_logger.setLevel(logging.INFO)
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    caplog_logger.addHandler(handler)
    try:
        _log_dgsm_progress("dynamic_dgsm", 1, 3, 50)
    finally:
        caplog_logger.removeHandler(handler)
    assert len(records) == 1
    assert records[0].getMessage() == "[dynamic_dgsm] 3/50 samples"


def test_aquakin_package_logger_has_a_nullhandler():
    # Library hygiene: aquakin never configures logging for the application, so
    # nothing is emitted unless the caller opts in.
    handlers = logging.getLogger("aquakin").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)
