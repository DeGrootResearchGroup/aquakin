"""What trame serves for the widget: the built bundle and its stylesheet."""

from pathlib import Path

serve = {"__aquakin_flowsheet": str(Path(__file__).with_name("serve").resolve())}
scripts = ["__aquakin_flowsheet/aquakin-flowsheet.umd.js"]
styles = ["__aquakin_flowsheet/aquakin-flowsheet.css"]
vue_use = ["aquakin_flowsheet"]
