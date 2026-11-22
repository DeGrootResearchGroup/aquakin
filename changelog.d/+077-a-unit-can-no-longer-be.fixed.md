A unit can no longer be named after one of its plant's models (for example a
splitter named `asm1`), which made `plant.parameter_index("asm1.muH")` return
the splitter's flow setpoint instead of the kinetic rate. Unit, port and
influent names containing `.`, `->` or whitespace are refused too, since
endpoints are written `unit.port`. Both raise `WiringError`.
