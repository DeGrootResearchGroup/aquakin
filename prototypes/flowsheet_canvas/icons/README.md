# Flowsheet unit icons

A set of 23 minimal, geometric line icons for the unit operations in the
flowsheet canvas. They are drawn to read at 20 px in the palette and at
20–28 px in a node card's header, and they take their colour from the
surrounding text (`currentColor`), so a node can tint its icon with its accent
colour.

`contact_sheet.html` shows every icon at 20, 32 and 64 px on light and dark
backgrounds, plus node-header and palette previews; `contact_sheet.png` is a
rendering of it.

## Style rules

- `viewBox="0 0 24 24"`; no `width`/`height`, `<style>`, scripts, external
  references or text.
- Strokes only: the root `<svg>` sets `fill="none"`, `stroke="currentColor"`,
  `stroke-width="1.75"`, `stroke-linecap="round"`, `stroke-linejoin="round"`.
  Small solid accents (bubbles, solids, junction dots) use
  `fill="currentColor" stroke="none"`, sparingly.
- Draw within 2..22 (2 px padding), at most six shape elements per icon, and
  snap to the 0.5 grid so lines stay crisp.
- One shared vocabulary across the set:
  - **Tank**: an open-topped rounded U (`M4 5v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V5`)
    for every tank-like unit (CSTR, MBR, IFAS, MBBR, SBR, storage).
  - **Hopper**: the same open top with a sloped bottom for settling units.
  - **Bubbles**: solid dots of radius 1.5 in a fixed rising cluster = aeration.
  - **Carriers**: hollow rings of radius 2 = biofilm media.
  - **Solids**: smaller solid dots (radius 1.25) = settled sludge.
  - **Arrow head**: a right-angle chevron with 2.5-unit legs (2-unit legs where
    space is tight).
  - **Junction**: a solid dot where streams meet or divide.
  - **Shaft**: a vertical line from above the rim = a mechanical mixer.

## Icons

Reactors

- `cstr.svg`: tank with a mixer shaft and a flat impeller blade.
- `cstr-aerated.svg`: the same tank, with the bubble cluster instead of the mixer.
- `mbr.svg`: tank with two submerged membrane plates (rounded slabs).
- `ifas.svg`: tank with two carriers above the bubble cluster (media plus aeration).
- `mbbr.svg`: tank with three carriers and no bubbles.
- `sbr.svg`: tank with a three-quarter cycle arrow (fill, react, settle, draw).
- `digester.svg`: egg-shaped vessel with a liquid level and a gas line leaving the dome.

Separators

- `clarifier-secondary.svg`: hopper with three stacked layer lines (the layered settling model).
- `clarifier-primary.svg`: the same hopper with a single downward settling arrow.
- `thickener.svg`: deeper hopper with a supernatant line and packed solids at the bottom.

Flow handling

- `mixer.svg`: two curved streams and a straight one meeting at a junction, leaving as one.
- `splitter.svg`: one stream reaching a junction and dividing into two arrowed branches.
- `splitter-setpoint.svg`: the splitter with a target ring (setpoint) in place of the junction dot.
- `splitter-threshold.svg`: the splitter with a dotted upper branch, the overflow that only carries flow above a threshold.
- `delay.svg`: a pipe (two parallel lines) carrying a train of chevrons, i.e. plug flow.
- `storage.svg`: tank with a level line and an inverted-triangle level marker.

Treatment and dosing

- `dosing.svg`: a drop falling onto a line.
- `uv.svg`: a lamp tube radiating over a channel floor.
- `chlorine-contact.svg`: plan view of a basin with two alternating baffles (serpentine path).

Control and boundaries

- `controller.svg`: a controller block inside a closed feedback loop with a measurement dot.
- `influent.svg`: an arrow entering an open bracket (into the plant).
- `effluent.svg`: an arrow leaving an open bracket (out of the plant).
- `translator.svg`: two state-model blocks with arrows running each way between them.

## Licence

These icons are original drawings made for this repository; they are not
traced from or based on any commercial product's icons. They are released with
the repository under the MIT licence.
