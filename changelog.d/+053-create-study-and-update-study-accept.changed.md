`create_study` and `update_study` accept an absolute `flowsheet:` path and
store it relative to the study folder (leaving it out when it is the
project's `flowsheet.yaml`). Study files on disk keep their names and format.
