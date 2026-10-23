// Supplementary Fig. 2: our network on 30-Hz video. Build:
// typst compile figures/figs2.typ (from manuscript/).
// Panels: layout.toml [figs2], all drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as figs2/<name>.svg; the rest are drawn as placeholders.
#let files = ("a", "b")

#let body = assemble("figs2", files: files)

#set page(width: 89mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
