// Fig. 4: Generalization and its limits. Build: typst compile figures/fig4.typ (from manuscript/).
// Panels: layout.toml [fig4], all drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as fig4/<name>.svg; the rest are drawn as placeholders.
#let files = ("a", "b", "c", "d", "e")

#let body = assemble("fig4", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
