The ASM1↔ADM1 translators (`ASM1toADM1`, `ADM1toASM1`) read the species
units the connected models declare and convert them into the units their
equations use, instead of assuming ASM1 in g/m3 and ADM1 in kg/m3 and
kmol/m3. A model declared on another basis (an ADM1 in g/m3) now translates
correctly. A declared unit that cannot be converted to the interface's (for
example inorganic N as a mass where moles are expected, or a model left at
the schema's default `mol/L` species units) now raises when the translator
is built; a unit that cannot be read gives a warning and is taken to be the
interface's. `Plant.connect` rejects a translator built for models that
declare different units from the models it connects. Results with the
shipped models are unchanged.
