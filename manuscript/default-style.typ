/// The draft: A4, serif text, a centred title block and the abstract as a section.
#let preprint(
  title: none,
  short-title: none,
  manuscript: "Preprint",
  authors: (),
  affiliations: (:),
  abstract: none,
  keywords: (),
  line-numbers: false,
  logo: false,
  body,
) = {
  set document(title: title, author: authors.map(a => a.name))
  set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
  set text(font: ("Libertinus Serif", "New Computer Modern"), size: 11pt, lang: "en")
  set par(justify: true, leading: 0.75em, spacing: 1.2em)
  set heading(numbering: none)
  show heading.where(level: 1): set text(size: 13pt)
  show heading.where(level: 2): set text(size: 11pt, style: "italic")
  set figure(supplement: [Fig.])
  set figure.caption(separator: [ | ])
  show figure.caption: set text(size: 9pt)
  show figure.caption: set align(left)
  show figure.where(kind: table): set figure.caption(position: top)
  set bibliography(style: "nature")

  // Affiliations in order of first use.
  let order = ()
  for a in authors {
    for key in a.at("affiliations", default: ()) {
      if key not in order { order.push(key) }
    }
  }
  align(center)[
    #text(size: 16pt, weight: "bold", title)
    #v(0.6em)
    #authors.map(a => [#a.name#super(a.at("affiliations", default: ()).map(k => str(order.position(x => x == k) + 1)).join(","))]).join[, ] \
    #text(size: 9.5pt, order.enumerate().map(((i, k)) => [#super(str(i + 1))#affiliations.at(k)]).join[ \ ])
  ]
  v(1em)
  if abstract != none [
    = Abstract

    #abstract
  ]
  body
}

/// The bibliography as it is: `#references(bibliography("refs.bib"))`.
#let references(bib) = bib

/// Supplementary figures and tables are numbered from 1 again.
#let supplement(body) = {
  counter(figure.where(kind: image)).update(0)
  counter(figure.where(kind: table)).update(0)
  body
}
