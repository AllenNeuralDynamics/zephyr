"""``lists``: the dataset's clip list, ``sessions.toml``, one entry per fetched session.

Entries are thermistor only (``thermistor = "thermistor/<session_id>.parquet"``, no
video): the whole session, read by ``zephyr annotate events`` and the QC. The header
(``[preprocess]``, ``[events]``) is rewritten from the spec. Entries are only added, so
marks, spans and reviews survive, but each one's ``group`` is (re)set to its subject
from the session metadata (never parsed from the session name, which can predate a
correction), so folds can hold out animals.
"""

from pathlib import Path

import tomlkit

from zephyr.config import Clip, ClipList, EventsSource, PreprocessParams

from . import storage
from .config import DatasetSpec, Layout, session_of

HEADER = [
    "Clip list of a zephyr-dataset folder: one thermistor-only entry per whole",
    "session. Written by `zephyr-dataset lists`; review marks and spans with",
    "`zephyr annotate events` on this file. `group` is the subject from the session",
    "metadata.",
]


SMALL_HEADER = [
    "The sessions of clips.toml marked `small` in folds.parquet, for small experiments:",
    "written by `zephyr-dataset folds`, `clips` and `boxes publish` (never edit by hand).",
]


def header(spec: DatasetSpec) -> dict:
    events = {"method": spec.events.method}
    if spec.events.params:
        events["params"] = tomlkit.inline_table()
        events["params"].update({k: float(v) for k, v in spec.events.params.items()})
    return {
        "preprocess": spec.preprocess.model_dump(mode="json", exclude_defaults=True),
        "events": events,
    }


def subjects(layout: Layout) -> dict[str, str]:
    """session_id -> subject_id, from the inventory (session metadata)."""
    inventory = storage.read_parquet(
        layout.inventory, columns=["session_id", "subject_id"]
    ).dropna()
    return dict(zip(inventory["session_id"], inventory["subject_id"].astype(str)))


def validate(doc, layout: Layout) -> list[Clip]:
    """The clips of *doc*, a list at the root of *layout*. A local list is checked whole
    by zephyr (its files must exist); a remote one header and entry by entry, as zephyr
    cannot see remote files."""
    data = doc.unwrap()
    if layout.local:
        return ClipList.model_validate(data, context={"base": Path(layout.root)}).clip
    PreprocessParams.model_validate(data.get("preprocess", {}))
    EventsSource.model_validate(data.get("events", {}))
    context = {"base": Path.cwd()}  # names only: never opened
    return [Clip.model_validate(c, context=context) for c in data["clip"]]


def write_small(layout: Layout) -> int:
    """``clips-small.toml``: the entries of ``clips.toml`` (boxes, marks and all) of the
    sessions ``folds.parquet`` marks ``small``; removed when there are none."""
    folds = storage.read_parquet(layout.folds)
    keep = set(folds.loc[folds["small"].fillna(False), "session_id"])
    doc = tomlkit.parse(layout.clip_list.read_text(encoding="utf-8"))
    entries = [e for e in doc["clip"] if str(e["group"]) in keep]
    if not entries:
        if layout.clip_list_small.exists():
            storage.delete([layout.clip_list_small])
        return 0
    out = tomlkit.document()
    for line in SMALL_HEADER:
        out.add(tomlkit.comment(line))
    out.add(tomlkit.nl())
    for key in ("preprocess", "events"):
        if key in doc:
            out.add(key, doc[key])
    out.add("clip", tomlkit.aot())
    for entry in entries:
        out["clip"].append(entry)
    validate(out, layout)
    storage.write_text(layout.clip_list_small, tomlkit.dumps(out))
    return len(entries)


def run(spec: DatasetSpec) -> int:
    layout = Layout(spec.root)
    path = layout.session_list
    subject = subjects(layout)
    if path.exists():
        doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    else:
        doc = tomlkit.document()
        for line in HEADER:
            doc.add(tomlkit.comment(line))
        doc.add(tomlkit.nl())
    if "derive" in doc:  # entries name their thermistor; nothing is derived
        del doc["derive"]
    for name, table in header(spec).items():
        if name in doc:
            doc[name] = table
        else:
            doc.add(name, table)
    if "clip" not in doc:
        doc.add("clip", tomlkit.aot())
    present = {session_of(c["thermistor"]) for c in doc["clip"]}
    added = 0
    for session_id in layout.sessions():
        if session_id in present:
            continue
        entry = tomlkit.table()
        entry.add("thermistor", f"thermistor/{layout.thermistor(session_id).name}")
        doc["clip"].append(entry)
        added += 1
    if not doc["clip"]:
        raise SystemExit("lists: no fetched sessions yet (run fetch first)")
    unknown = []
    for entry in doc["clip"]:
        session_id = session_of(entry["thermistor"])
        if session_id not in subject:
            unknown.append(session_id)
        elif str(entry.get("group", "")) != subject[session_id]:
            entry["group"] = subject[session_id]
    if unknown:
        raise SystemExit(f"lists: no subject in the inventory for {sorted(unknown)}")
    validate(doc, layout)
    storage.write_text(path, tomlkit.dumps(doc))
    print(f"lists: {len(doc['clip'])} sessions ({added} added) in {path.name}")
    return added
