A reaction can declare `units_acknowledged: "<reason>"` for a rate whose
units are knowingly unconventional. `check_units()` then leaves its findings
out; `check_units(include_acknowledged=True)` still shows them.
