// Fig. 2: The corpus. Build: typst compile figures/fig2.typ (from manuscript/).
// Panels: layout.toml [fig2], all drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as fig2/<name>.svg; the rest are drawn as placeholders.
#let files = ("a", "b", "c")

#let body = assemble("fig2", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
