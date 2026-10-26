// Fig. 5: Using it on new data. Build: typst compile figures/fig5.typ (from manuscript/).
// Panels: layout.toml [fig5], drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as fig5/<name>.svg; the rest are drawn as placeholders.
#let files = ("a", "b", "c")

#let body = assemble("fig5", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
