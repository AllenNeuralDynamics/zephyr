// Supplementary Fig. 7: refining on a different behavioural task. Build:
// typst compile figures/figs7.typ (from manuscript/).
// Panels: layout.toml [figs7], all drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as figs7/<name>.svg; the rest are drawn as placeholders.
#let files = ("a",)

#let body = assemble("figs7", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
