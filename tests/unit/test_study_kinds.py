"""Every study kind is complete: a runner, its required fields, a form entry and a
result reader, all keyed by the one list of kinds ``study.yaml`` accepts.

The view of each kind's results lives in ``aquakin_ui`` and is pinned with the
worker protocol in ``tests/ui/test_protocol.py``.
"""

import pytest

from aquakin.study import StudyError
from aquakin.study import form as study_form_module
from aquakin.study import study as study_module


def _gaps(kinds) -> dict[str, list[str]]:
    """What each kind lacks, by the registry it is missing from."""
    registries = {
        "runner": study_module._RUNNERS,
        "required fields": study_module._REQUIRED,
        "form": study_form_module._KINDS,
        "reader": study_form_module._READERS,
    }
    gaps = {
        kind: [name for name, registry in registries.items() if kind not in registry]
        for kind in kinds
    }
    return {kind: missing for kind, missing in gaps.items() if missing}


def test_every_kind_has_a_runner_required_fields_a_form_and_a_reader():
    assert _gaps(study_module.KINDS) == {}
    # No registry holds a kind study.yaml would refuse.
    for registry in (
        study_module._RUNNERS,
        study_module._REQUIRED,
        study_form_module._KINDS,
        study_form_module._READERS,
    ):
        assert set(registry) == set(study_module.KINDS)


def test_the_kinds_are_the_ones_study_yaml_accepts():
    from typing import get_args

    accepted = get_args(study_module._Study.model_fields["kind"].annotation)
    assert set(accepted) == set(study_module.KINDS)


def test_a_new_kind_without_its_pieces_is_reported():
    """The coverage check names each missing piece of a kind nobody has finished."""
    assert _gaps((*study_module.KINDS, "survey")) == {
        "survey": ["runner", "required fields", "form", "reader"]
    }


def test_an_unknown_kind_has_no_reader_rather_than_the_last_one():
    with pytest.raises(StudyError, match="no result reader for a 'survey' study"):
        study_form_module._reader("survey")


def test_result_file_names_are_the_stored_ones():
    """Existing projects keep reading: the constants spell the names on disk."""
    s = study_module
    assert (
        s.PARAMETERS_FILE,
        s.FINAL_STATE_FILE,
        s.STREAMS_STEADY_FILE,
        s.MEASURED_STEADY_FILE,
        s.STREAMS_FILE,
        s.MEASURED_FILE,
        s.FISHER_FILE,
        s.FISHER_SUMMARY_FILE,
        s.FIT_FILE,
        s.FITTED_FILE,
        s.COVARIANCE_FILE,
        s.PROFILE_FILE,
        s.PROFILE_SUMMARY_FILE,
        s.RANKING_FILE,
        s.STUDIES_DIR,
        s.STUDY_FILE,
        s.SPEC_FILE,
    ) == (
        "parameters.yaml",
        "final_state.yaml",
        "streams.yaml",
        "measured.yaml",
        "streams.csv",
        "measured.csv",
        "fisher.csv",
        "summary.yaml",
        "fit.yaml",
        "fitted.csv",
        "covariance.csv",
        "profile.csv",
        "profile.yaml",
        "ranking.csv",
        "studies",
        "study.yaml",
        "flowsheet.yaml",
    )


@pytest.mark.parametrize(
    ("state", "has_results"),
    [
        ("not run", False),
        ("failed", False),
        ("cancelled", False),
        ("fresh", True),
        ("stale", True),
        ("damaged", True),
    ],
)
def test_which_states_have_results(state, has_results):
    assert state in study_module.STATES
    assert study_module.StudyStatus(state, frozen=False).has_results is has_results
