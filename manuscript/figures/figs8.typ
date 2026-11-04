// Supplementary Fig. 8: sessions without a thermistor from every camera. Build:
// typst compile figures/figs8.typ (from manuscript/).
// Panels: layout.toml [figs8], all drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as figs8/<name>.svg; the rest are drawn as placeholders.
#let files = ("a",)

#let body = assemble("figs8", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
