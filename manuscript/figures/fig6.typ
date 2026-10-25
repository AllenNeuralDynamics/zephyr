// Fig. 6: When to trust a prediction. Build: typst compile figures/fig6.typ (from manuscript/).
// Panels: layout.toml [fig6], drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as fig6/<name>.svg; the rest are drawn as placeholders.
#let files = ("a",)

#let body = assemble("fig6", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
