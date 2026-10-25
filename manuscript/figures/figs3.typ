// Supplementary Fig. 3: our network on each camera's video. Build:
// typst compile figures/figs3.typ (from manuscript/).
// Panels: layout.toml [figs3], all drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as figs3/<name>.svg; the rest are drawn as placeholders.
#let files = ("a", "b", "c")

#let body = assemble("figs3", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
