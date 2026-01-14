import sys
import math
import re
import csv
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QFile, Qt
from PySide6.QtGui import QKeySequence, QShortcut, QStandardItem, QStandardItemModel
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtWidgets import QHeaderView


# ---------------------------
# Data model + CSV loading
# ---------------------------

@dataclass
class Fighter:
    data: Dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.data[name]
        except KeyError as e:
            raise AttributeError(f"No such attribute: {name}") from e


def _to_number_if_possible(value: str) -> Any:
    if value is None:
        return None
    s = value.strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def load_fighters_from_csv(path: str) -> List[Fighter]:
    fighters: List[Fighter] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row.")
        for row in reader:
            cleaned = {k: _to_number_if_possible(v) for k, v in row.items()}
            fighters.append(Fighter(cleaned))
    return fighters


# ---------------------------
# Helpers
# ---------------------------

def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _get_field_value(f: Fighter, field: str) -> Any:
    if not field:
        return None

    if field in f.data:
        return f.data[field]

    target = _norm_key(field)
    for k in f.data.keys():
        if _norm_key(k) == target:
            return f.data[k]

    return None


def fighter_name(f: Fighter) -> str:
    v = _get_field_value(f, "name")
    if v is None:
        v = _get_field_value(f, "Name")
    return str(v) if v is not None else "<unknown>"


def set_status(ui, msg: str):
    try:
        ui.statusbar.showMessage(msg)
    except Exception:
        pass


def _try_parse_float(s: str) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip()
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _string_similarity(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _numeric_similarity(a: float, b: float) -> float:
    return 1.0 / (1.0 + abs(a - b))


def build_headers_all_columns(fighters: List[Fighter]) -> List[str]:
    """score + all CSV columns (stable order based on first row, then extras)."""
    if not fighters:
        return ["score"]

    first = list(fighters[0].data.keys())
    all_keys = set(first)
    for f in fighters[1:]:
        all_keys.update(f.data.keys())

    extras = [k for k in sorted(all_keys, key=lambda s: s.lower()) if k not in first]
    return ["score"] + first + extras


def fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


# ---------------------------
# Search (string + numeric)
# ---------------------------

def _parse_search(query: str, default_field: str = "name") -> Tuple[str, str]:
    q = query.strip()
    m = re.match(r"^\s*\[(?P<field>[^\]]+)\]\s*:\s*(?P<value>.+?)\s*$", q)
    if m:
        return m.group("field").strip(), m.group("value").strip()
    return default_field, q


def sort_fighters_by_query(
    fighters: List[Fighter],
    query: str,
    default_field: str = "name",
) -> List[Tuple[float, Fighter]]:
    field, raw_value = _parse_search(query, default_field=default_field)
    q_num = _try_parse_float(raw_value)

    scored: List[Tuple[float, Fighter]] = []
    for f in fighters:
        fv = _get_field_value(f, field)
        if q_num is not None and isinstance(fv, (int, float)):
            score = _numeric_similarity(q_num, float(fv))
        else:
            score = _string_similarity(str(raw_value), "" if fv is None else str(fv))
        scored.append((score, f))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


# ---------------------------
# Skill match
# ---------------------------

def _parse_feature_spec(spec: Optional[str]) -> List[Tuple[str, float]]:
    # default
    if not spec or spec.strip() == "":
        return [("win rate", 1.0), ("weight", 1.0)]

    s = spec.strip()
    m = re.match(r"^\s*\[(.*)\]\s*$", s)
    inside = m.group(1).strip() if m else s

    if not inside:
        return [("win rate", 1.0), ("weight", 1.0)]

    parts = [p.strip() for p in inside.split(",") if p.strip()]
    out: List[Tuple[str, float]] = []
    for p in parts:
        mm = re.match(r"^(.*?)\s*\*\s*([0-9]*\.?[0-9]+)\s*$", p)
        if mm:
            feat = mm.group(1).strip()
            w = float(mm.group(2))
            out.append((feat, w))
        else:
            out.append((p, 1.0))
    return out


def _column_stats_numeric(fighters: List[Fighter], feature: str) -> Tuple[float, float]:
    vals = []
    for f in fighters:
        v = _get_field_value(f, feature)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return (0.0, 1.0)
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / max(1, (len(vals) - 1))
    std = math.sqrt(var) or 1.0
    return (mean, std)


def _find_best_match_target(fighters: List[Fighter], target_query: str) -> Optional[Fighter]:
    scored = sort_fighters_by_query(fighters, target_query, default_field="name")
    return scored[0][1] if scored else None


def match_by_skill(
    fighters: List[Fighter],
    target_query: str,
    feature_spec: Optional[str] = None,
    top_n: int = 25,
) -> List[Tuple[float, Fighter]]:
    target = _find_best_match_target(fighters, target_query)
    if target is None:
        return []

    features = _parse_feature_spec(feature_spec)

    stats: Dict[str, Tuple[float, float]] = {}
    for feat, _w in features:
        stats[feat] = _column_stats_numeric(fighters, feat)

    def dist(a: Fighter, b: Fighter) -> float:
        d2 = 0.0
        used = 0
        for feat, w in features:
            va = _get_field_value(a, feat)
            vb = _get_field_value(b, feat)
            if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
                continue
            mean, std = stats[feat]
            za = (float(va) - mean) / std
            zb = (float(vb) - mean) / std
            d = za - zb
            d2 += (w * d) ** 2
            used += 1
        return float("inf") if used == 0 else math.sqrt(d2)

    scored: List[Tuple[float, Fighter]] = []
    for f in fighters:
        if f is target:
            continue
        d = dist(target, f)
        if d == float("inf"):
            continue
        score = 1.0 / (1.0 + d)
        scored.append((score, f))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:top_n]


# ---------------------------
# Full system auto-match config
# ---------------------------

def parse_full_match_config(text: str) -> Optional[str]:
    """
    Accepts:
      "" -> defaults
      "weight, win rate" -> "[weight, win rate]"
      "[win rate*2, weight]" -> as-is
    """
    t = (text or "").strip()
    if t == "":
        return None
    if "[" in t and "]" in t:
        return t
    parts = [p.strip() for p in t.split(",") if p.strip()]
    if not parts:
        return None
    return "[" + ", ".join(parts) + "]"


def build_full_match_model(
    fighters: List[Fighter],
    feature_spec: Optional[str],
    cap: int = 1500,
) -> QStandardItemModel:
    work = fighters[:cap] if len(fighters) > cap else fighters

    features = _parse_feature_spec(feature_spec)
    feat_names = [feat for feat, _w in features]

    headers = ["Fighter", "Best Match", "Score"]
    for feat in feat_names:
        headers.append(f"{feat} (A)")
        headers.append(f"{feat} (B)")

    model = QStandardItemModel(0, len(headers))
    model.setHorizontalHeaderLabels(headers)

    stats = {feat: _column_stats_numeric(work, feat) for feat in feat_names}

    def vec(f: Fighter) -> Optional[List[float]]:
        out = []
        used = 0
        for feat, w in features:
            v = _get_field_value(f, feat)
            if not isinstance(v, (int, float)):
                out.append(float("nan"))
                continue
            mean, std = stats[feat]
            out.append(w * ((float(v) - mean) / std))
            used += 1
        return None if used == 0 else out

    vectors = [vec(f) for f in work]

    for i, f in enumerate(work):
        vi = vectors[i]
        best_j = -1
        best_d2 = float("inf")

        if vi is not None:
            for j, g in enumerate(work):
                if i == j:
                    continue
                vj = vectors[j]
                if vj is None:
                    continue

                d2 = 0.0
                used = 0
                for a, b in zip(vi, vj):
                    if math.isnan(a) or math.isnan(b):
                        continue
                    d = a - b
                    d2 += d * d
                    used += 1
                if used == 0:
                    continue

                if d2 < best_d2:
                    best_d2 = d2
                    best_j = j

        best_f = work[best_j] if best_j != -1 else None
        score = (1.0 / (1.0 + math.sqrt(best_d2))) if best_j != -1 else 0.0

        row = [
            QStandardItem(fighter_name(f)),
            QStandardItem(fighter_name(best_f) if best_f else "<none>"),
            QStandardItem(f"{score:.3f}"),
        ]

        for feat in feat_names:
            row.append(QStandardItem(fmt_cell(_get_field_value(f, feat))))
            row.append(QStandardItem(fmt_cell(_get_field_value(best_f, feat)) if best_f else ""))

        model.appendRow(row)

    return model


# ---------------------------
# Table population
# ---------------------------

def build_results_model(headers: List[str], rows: List[Tuple[float, Fighter]]) -> QStandardItemModel:
    model = QStandardItemModel(0, len(headers))
    model.setHorizontalHeaderLabels(headers)

    for score, f in rows:
        items: List[QStandardItem] = []
        for h in headers:
            if h == "score":
                items.append(QStandardItem(f"{score:.3f}"))
            else:
                items.append(QStandardItem(fmt_cell(f.data.get(h))))
        model.appendRow(items)

    return model


def apply_table_niceness(view):
    # nice defaults for QTableView
    hdr = view.horizontalHeader()
    hdr.setStretchLastSection(True)
    hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
    view.setAlternatingRowColors(True)


# ---------------------------
# UI + Help
# ---------------------------

def load_ui(path: str):
    loader = QUiLoader()
    f = QFile(path)
    if not f.open(QFile.ReadOnly):
        raise RuntimeError(f"Could not open UI file: {path}")
    ui = loader.load(f)
    f.close()
    if ui is None:
        raise RuntimeError("Failed to load UI.")
    return ui


def show_help(ui):
    txt = (
        "SEARCH:\n"
        "  - Type into Search box, click Search.\n"
        "  - Examples:\n"
        "      James\n"
        "      [weight]:250\n"
        "      [games played]:300\n\n"
        "MATCH (skill match):\n"
        "  - Type a target fighter into Match box, click Match.\n"
        "  - Examples:\n"
        "      James\n"
        "      [id]:2\n\n"
        "FULL MATCH CONFIG:\n"
        "  - Optional config in FullMatchConfig.\n"
        "  - Blank = defaults (win rate, weight)\n"
        "  - Example:\n"
        "      weight, win rate\n"
        "    (also supports: [win rate*2, weight])\n\n"
        "SHORTCUTS:\n"
        "  - Ctrl+Enter in Search box: Search\n"
        "  - Ctrl+Enter in Match box: Match\n"
        "  - Ctrl+F: Search\n"
        "  - Ctrl+M: Match\n"
    )
    QMessageBox.information(ui, "Help", txt)


# ---------------------------
# Main
# ---------------------------

def main():
    app = QApplication(sys.argv)
    ui = load_ui("fighter.ui")
    ui.setWindowTitle("Fighter Matcher")

    try:
        fighters = load_fighters_from_csv("fighters.csv")
    except Exception as e:
        QMessageBox.critical(ui, "Error", f"Failed to load fighters.csv:\n{e}")
        ui.show()
        sys.exit(app.exec())

    headers = build_headers_all_columns(fighters)

    # init empty models
    ui.searchTable.setModel(build_results_model(headers, []))
    ui.matchTable.setModel(build_results_model(headers, []))
    ui.fullMatchTable.setModel(QStandardItemModel(0, 0))

    apply_table_niceness(ui.searchTable)
    apply_table_niceness(ui.matchTable)
    apply_table_niceness(ui.fullMatchTable)

    set_status(ui, f"Loaded {len(fighters)} fighters.")

    def do_search():
        query = ui.searchBox.toPlainText().strip()
        if not query:
            set_status(ui, "Search box is empty.")
            return

        scored = sort_fighters_by_query(fighters, query, default_field="name")[:200]
        ui.searchTable.setModel(build_results_model(headers, scored))
        apply_table_niceness(ui.searchTable)
        set_status(ui, f"Search complete: {len(scored)} results shown.")

    def do_match():
        query = ui.matchTextBox.toPlainText().strip()
        if not query:
            set_status(ui, "Match box is empty.")
            return

        scored = match_by_skill(fighters, target_query=query, feature_spec=None, top_n=200)
        ui.matchTable.setModel(build_results_model(headers, scored))
        apply_table_niceness(ui.matchTable)
        set_status(ui, f"Match complete: {len(scored)} results shown.")

    def do_full_match():
        cfg = ui.fullMatchConfig.toPlainText()
        feature_spec = parse_full_match_config(cfg)  # None -> defaults
        model = build_full_match_model(fighters, feature_spec=feature_spec, cap=1500)
        ui.fullMatchTable.setModel(model)
        apply_table_niceness(ui.fullMatchTable)

        spec_show = feature_spec if feature_spec else "[win rate, weight]"
        set_status(ui, f"Full match complete using {spec_show}.")

    ui.searchButton.clicked.connect(do_search)
    ui.matchButton.clicked.connect(do_match)
    ui.fullMatchButton.clicked.connect(do_full_match)
    ui.helpButton.clicked.connect(lambda: show_help(ui))

    # Shortcuts (QTextEdit eats Enter, so Ctrl+Enter)
    QShortcut(QKeySequence("Ctrl+Return"), ui.searchBox).activated.connect(do_search)
    QShortcut(QKeySequence("Ctrl+Enter"), ui.searchBox).activated.connect(do_search)
    QShortcut(QKeySequence("Ctrl+F"), ui).activated.connect(do_search)

    QShortcut(QKeySequence("Ctrl+Return"), ui.matchTextBox).activated.connect(do_match)
    QShortcut(QKeySequence("Ctrl+Enter"), ui.matchTextBox).activated.connect(do_match)
    QShortcut(QKeySequence("Ctrl+M"), ui).activated.connect(do_match)

    ui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
