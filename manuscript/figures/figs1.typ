// Supplementary Fig. 1: animal and rig as sources of score differences. Build:
// typst compile figures/figs1.typ (from manuscript/).
// Panels: layout.toml [figs1], all drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as figs1/<name>.svg; the rest are drawn as placeholders.
#let files = ("a",)

#let body = assemble("figs1", files: files)

#set page(width: 89mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
