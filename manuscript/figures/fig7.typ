// Fig. 7: An analysis video makes possible. Build: typst compile figures/fig7.typ (from manuscript/).
// Panels: layout.toml [fig7], drawn by notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as fig7/<name>.svg; the rest are drawn as placeholders.
#let files = ()

#let body = assemble("fig7", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
