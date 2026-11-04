// Supplementary Fig. 9: every quality-control measure. Build:
// typst compile figures/figs9.typ (from manuscript/).
// Panels: layout.toml [figs9], all drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as figs9/<name>.svg; the rest are drawn as placeholders.
#let files = ("a", "b")

#let body = assemble("figs9", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
