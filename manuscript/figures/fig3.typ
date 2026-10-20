// Fig. 3: The benchmark. Build: typst compile figures/fig3.typ (from manuscript/).
// Panels: layout.toml [fig3], all drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as fig3/<name>.svg; the rest are drawn as placeholders.
#let files = ("a", "b", "c", "d")

#let body = assemble("fig3", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
