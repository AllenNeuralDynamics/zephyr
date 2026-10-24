// Supplementary Fig. 4: what each camera sees. Build:
// typst compile figures/figs4.typ (from manuscript/).
// Panels: layout.toml [figs4], all drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as figs4/<name>.svg; the rest are drawn as placeholders.
#let files = ("a",)

#let body = assemble("figs4", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
