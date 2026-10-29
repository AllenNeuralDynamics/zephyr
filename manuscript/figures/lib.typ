// Assembles a figure from its panels, as laid out in layout.toml.
// Each panel is <figure>/<panel>.svg at its printed size; a missing one is drawn as a
// labelled placeholder of the same size, so the layout can be judged before it exists.

#let layout = toml("layout.toml")

#let letter(name) = text(font: "Arial", size: 8pt, weight: "bold")[#name]

#let placeholder(spec) = rect(
  width: 100%, height: 100%, fill: luma(246),
  stroke: (paint: luma(160), dash: "dashed", thickness: 0.5pt), inset: 4pt,
)[
  #set text(font: "Arial", size: 6pt, fill: luma(90))
  #align(center + horizon)[#spec.what \ _(#spec.kind, #spec.width_mm × #spec.height_mm mm)_]
]

#let panel(figure, name, files) = {
  let spec = layout.at(figure).panels.at(name)
  let w = spec.width_mm * 1mm
  let h = spec.height_mm * 1mm
  box(width: w, height: h)[
    #if name in files { image(figure + "/" + name + ".svg", width: w, height: h) } else { placeholder(spec) }
    #place(top + left, dx: -2.5mm, dy: -1mm, letter(name))
  ]
}

// `files`: the panel names that exist as SVG (Typst cannot test for a file itself).
#let assemble(figure, files: ()) = {
  let spec = layout.at(figure)
  let gap = spec.gap_mm * 1mm
  set block(spacing: gap)
  // A row item is a panel name, or a list of names stacked top to bottom.
  let item(entry) = if type(entry) == array {
    stack(dir: ttb, spacing: gap, ..entry.map(name => panel(figure, name, files)))
  } else { panel(figure, entry, files) }
  for row in spec.rows {
    block(stack(dir: ltr, spacing: gap, ..row.map(item)))
  }
}
