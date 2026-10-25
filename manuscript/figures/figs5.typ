// Supplementary Fig. 5: timing of slow breaths. Build:
// typst compile figures/figs5.typ (from manuscript/).
// Panels: layout.toml [figs5], all drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as figs5/<name>.svg; the rest are drawn as placeholders.
#let files = ("a", "b", "c", "d")

#let body = assemble("figs5", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
