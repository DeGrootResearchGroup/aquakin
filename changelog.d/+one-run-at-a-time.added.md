A study runs once at a time, whichever program starts it. While it runs, its
folder holds `running.yaml` with when the run started and its progress, so
`aquakin study check`, `aquakin.study.check_study` (`StudyStatus.running`) and
the flowsheet editor all show a run in progress, wherever it was started. A
second run of the study is refused, and so are freezing, unfreezing and
editing it. `aquakin study cancel` (and `aquakin.study.cancel_study`) stops a
run at its next checkpoint from any program; the editor's Cancel works on runs
other programs started. A run that crashes or is killed leaves no lock behind.
Studies an AI assistant runs through `aquakin mcp` are separate processes and
go on after the assistant's app closes.
