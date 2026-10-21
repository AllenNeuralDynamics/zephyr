// Fig. 1: Measuring breathing from face video. Build: typst compile figures/fig1.typ
// (from manuscript/). Panels: layout.toml [fig1]; data panels from
// notebooks/manuscript_figures.py, schematics drawn by hand (a: the rig rendering from
// 3d-vr-foraging, renders/figure/FIG1_overview.svg).

#import "lib.typ": assemble

// Panels that exist as fig1/<name>.svg; the rest are drawn as placeholders.
#let files = ("a", "c", "d", "e")

#let body = assemble("fig1", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
