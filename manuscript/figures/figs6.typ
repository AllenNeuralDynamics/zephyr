// Supplementary Fig. 6: where and when the network looks. Build:
// typst compile figures/figs6.typ (from manuscript/).
// Panels: layout.toml [figs6], all drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as figs6/<name>.svg; the rest are drawn as placeholders.
#let files = ("a", "b", "c")

#let body = assemble("figs6", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
