"""Compare every model on one common, labelled dataset, ready for the report.

    python -m ppe.evaluate --data data.yaml --root D:/datasets/ppe-combined
    python -m ppe.evaluate --data data.yaml -w models/best-apron.pt model-versions/yolo11s-v6.pt
    python -m ppe.evaluate --data data.yaml --report internship-report.md   # also fill the report

Training metrics cannot rank the model versions: each was scored on its own validation split,
and some on different datasets altogether. This tool scores all of them on the same images,
the dataset's ``test`` split by default, and reports three things for each model:

**Detection.** Per-class precision, recall, AP@0.5 and AP@0.5:0.95, matched by class *name*.
Models do not share class indices (best-apron puts ``person`` first), and some lack classes
(only best-apron has ``Apron``). A model is never scored on a class it cannot detect; mAP is
also given over the classes every model has, which is the like-for-like figure.

**Decision.** What the station would have done. Each image is run through the station's own
logic (subject filter, containment, counts, any_of groups, forbidden classes) twice: once on
the ground-truth boxes, once on the model's detections at the configured confidence floors.
Disagreements are counted in the two directions that matter (report Table 1):
  false permit  the station would allow the machine, the labels say PPE is missing
  false stop    the station would stop the machine, the labels say the worker is compliant
This is a single-frame verdict, without the hold windows and debounce that smooth it in
operation. It measures the model, not the timing.

**Speed.** Median time per frame through the station's detector on this machine.

Output: ``model-evaluation.md`` (Markdown tables, captioned for the report), ``.csv`` and
``.json`` in --out. With --report, the summary and detail tables also replace the marked
blocks in the report source, ready for ``python docs/report/build_docx.py``.

The images are photographs of people at work. Everything here runs locally; nothing is
uploaded, and only aggregate numbers are written.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .config import Config, ModelCfg
from .detector import Detection

log = logging.getLogger("ppe.evaluate")

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Verdicts, as strings so they print and serialise without the tower module.
OK, VIOLATION, STANDBY, DEGRADED = "ok", "violation", "standby", "degraded"

MARK = "<!-- evaluation:{} -->"
END = "<!-- /evaluation:{} -->"


# -- dataset ------------------------------------------------------------------


@dataclass
class Sample:
    image: Path
    truth: list[Detection]  # ground truth, conf 1.0, source-frame pixels


@dataclass
class Dataset:
    names: list[str]  # class names in the dataset's own index order
    samples: list[Sample]
    split: str


def _names(raw) -> list[str]:
    if isinstance(raw, dict):
        return [raw[k] for k in sorted(raw, key=int)]
    return list(raw)


def _image_list(root: Path, entry) -> list[Path]:
    """Images for a split entry: a directory, a .txt list, or a list of either."""
    entries = entry if isinstance(entry, list) else [entry]
    out: list[Path] = []
    for e in entries:
        p = Path(e)
        p = p if p.is_absolute() else root / p
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
        elif p.suffix == ".txt" and p.is_file():
            for line in p.read_text().splitlines():
                if line.strip():
                    q = Path(line.strip())
                    out.append(q if q.is_absolute() else root / q)
        else:
            raise FileNotFoundError(f"split entry not found: {p}")
    return out


def _label_path(image: Path) -> Path:
    """YOLO convention: the last /images/ in the path becomes /labels/, suffix .txt."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _image_size(path: Path) -> tuple[int, int]:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return img.shape[1], img.shape[0]


def read_labels(path: Path, names: list[str], size: tuple[int, int]) -> list[Detection]:
    """One YOLO label file as pixel-space Detections (missing file = no objects)."""
    if not path.is_file():
        return []
    w, h = size
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
        out.append(
            Detection(
                names[cls],
                1.0,
                ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h),
            )
        )
    return out


def load_dataset(
    data: Path, root: Path | None = None, split: str = "test", limit: int = 0
) -> Dataset:
    import yaml

    spec = yaml.safe_load(data.read_text())
    names = _names(spec["names"])
    base = Path(root) if root else Path(spec.get("path") or data.parent)
    if not base.is_absolute():
        base = data.parent / base
    if split not in spec:
        raise KeyError(
            f"{data} has no '{split}' split "
            f"(has: {[k for k in ('train', 'val', 'test') if k in spec]})"
        )
    images = _image_list(base, spec[split])
    if limit:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"no images in the '{split}' split under {base}")
    samples = [
        Sample(img, read_labels(_label_path(img), names, _image_size(img))) for img in images
    ]
    return Dataset(names, samples, split)


# -- detection metrics ----------------------------------------------------------


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4), xyxy."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(preds: list[Detection], truth: list[Detection]) -> np.ndarray:
    """True-positive matrix (n_preds, 10): each prediction, at each IoU threshold.

    Greedy in confidence order, one ground-truth box per prediction, same class only —
    the COCO rule. A duplicate box on an already-matched object is a false positive.
    """
    tp = np.zeros((len(preds), len(IOU_THRESHOLDS)), dtype=bool)
    if not preds or not truth:
        return tp
    order = sorted(range(len(preds)), key=lambda i: -preds[i].conf)
    iou = iou_matrix(np.array([p.xyxy for p in preds]), np.array([t.xyxy for t in truth]))
    same = np.array([[p.name == t.name for t in truth] for p in preds])
    iou = np.where(same, iou, 0.0)
    for k, thr in enumerate(IOU_THRESHOLDS):
        taken = np.zeros(len(truth), dtype=bool)
        for i in order:
            cand = np.where(~taken & (iou[i] >= thr))[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                taken[j] = True
                tp[i, k] = True
    return tp


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision envelope at 101 recall points, as Ultralytics computes it.

    The same arithmetic as the training metrics, so the two can be compared: the curve is
    closed at (recall 1, precision 0) and interpolated linearly, which caps a perfect
    detector at 0.995 and credits a model whose recall stops short with the triangle to
    (1, 0). Rankings are unaffected; absolute values match the training records.
    """
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([1.0], precision, [0.0]))
    p = np.flip(np.maximum.accumulate(np.flip(p)))
    grid = np.linspace(0, 1, 101)
    integrate = getattr(np, "trapezoid", None) or np.trapz  # renamed in NumPy 2
    return float(integrate(np.interp(grid, r, p), grid))


@dataclass
class ClassScore:
    name: str
    instances: int
    detectable: bool
    precision: float = float("nan")
    recall: float = float("nan")
    ap50: float = float("nan")
    ap: float = float("nan")  # AP@0.5:0.95


def score_classes(
    names: list[str],
    detectable: set[str],
    tps: list[np.ndarray],
    preds: list[list[Detection]],
    truth: list[list[Detection]],
) -> list[ClassScore]:
    """Per-class P, R (at the confidence that maximises mean F1 over classes), AP50, AP."""
    tp = np.concatenate(tps) if tps else np.zeros((0, 10), bool)
    conf = np.array([p.conf for ps in preds for p in ps])
    pcls = np.array([p.name for ps in preds for p in ps])
    counts = {n: sum(t.name == n for ts in truth for t in ts) for n in names}
    grid = np.linspace(0, 1, 1000)
    curves, scores = {}, []
    for n in names:
        s = ClassScore(n, counts[n], n in detectable)
        if not s.detectable or counts[n] == 0:
            scores.append(s)
            continue
        idx = np.where(pcls == n)[0]
        idx = idx[np.argsort(-conf[idx])]
        ctp = np.cumsum(tp[idx], axis=0)
        cfp = np.cumsum(~tp[idx], axis=0)
        rec = ctp / counts[n]
        prec = ctp / np.maximum(ctp + cfp, 1)
        aps = [average_precision(rec[:, k], prec[:, k]) if len(idx) else 0.0 for k in range(10)]
        s.ap50, s.ap = aps[0], float(np.mean(aps))
        # P and R as functions of the confidence threshold, for the shared operating point.
        if len(idx):
            c = conf[idx]
            curves[n] = (
                np.interp(-grid, -c, rec[:, 0], left=0),
                np.interp(-grid, -c, prec[:, 0], left=1),
            )
        else:
            curves[n] = (np.zeros_like(grid), np.ones_like(grid))
        scores.append(s)
    if curves:
        f1 = np.mean([2 * p * r / (p + r + 1e-9) for r, p in curves.values()], axis=0)
        best = int(np.argmax(f1))
        for s in scores:
            if s.name in curves:
                s.recall, s.precision = curves[s.name][0][best], curves[s.name][1][best]
    return scores


# -- decision metrics -----------------------------------------------------------


def verdict(dets: list[Detection], cfg: Config, unavailable: Iterable[str] = ()) -> str:
    """The station's single-frame verdict on one image's detections."""
    from .subject import focus
    from .tower import ComplianceMonitor

    ppe = cfg.ppe
    # No debounce: one frame, judged on its own.
    zero = dict.fromkeys(("ok", "violation", "standby", "degraded"), 0.0)
    monitor = ComplianceMonitor(replace(ppe, confirm_sec=zero), list(unavailable))
    f = focus(dets, ppe.subject, ppe.containment, ppe.containment_map())
    return monitor.update([f.accepted], t=1.0).value


def with_floors(dets: list[Detection], cfg: Config) -> list[Detection]:
    """Keep what the station keeps: each class above its own configured floor."""
    floors = {c.name: (c.conf if c.conf is not None else cfg.model.conf) for c in cfg.ppe.classes}
    return [d for d in dets if d.name in floors and d.conf >= floors[d.name]]


@dataclass
class Decisions:
    images: int = 0
    agree: int = 0
    false_permit: int = 0  # model OK, labels VIOLATION
    false_stop: int = 0  # model VIOLATION/STANDBY, labels OK
    truth_ok: int = 0
    truth_violation: int = 0
    truth_standby: int = 0
    other: int = 0  # any other disagreement (e.g. STANDBY vs VIOLATION)
    # Images the labels call compliant only through a class this model cannot detect (an
    # apron worn instead of sleeves, judged by a model without an Apron class). The model
    # must stop them whatever its detection quality, so they are counted apart.
    truth_ok_unseeable: int = 0
    false_stop_unseeable: int = 0
    # (image, labels' verdict, model's verdict) for every image they disagree on
    disagreements: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def false_permit_rate(self) -> float:
        return self.false_permit / self.truth_violation if self.truth_violation else float("nan")

    @property
    def false_stop_rate(self) -> float:
        return self.false_stop / self.truth_ok if self.truth_ok else float("nan")

    @property
    def false_stops_seeable(self) -> int:
        return self.false_stop - self.false_stop_unseeable

    @property
    def truth_ok_seeable(self) -> int:
        return self.truth_ok - self.truth_ok_unseeable

    @property
    def false_stop_rate_seeable(self) -> float:
        """False stops on the compliant images this model could have judged compliant."""
        n = self.truth_ok_seeable
        return self.false_stops_seeable / n if n else float("nan")

    def add(self, truth: str, pred: str, image: str = "", unseeable: bool = False) -> None:
        self.images += 1
        self.truth_ok += truth == OK
        self.truth_violation += truth == VIOLATION
        self.truth_standby += truth == STANDBY
        self.truth_ok_unseeable += truth == OK and unseeable
        if truth == pred:
            self.agree += 1
            return
        self.disagreements.append((image, truth, pred))
        if pred == OK and truth == VIOLATION:
            self.false_permit += 1
        elif truth == OK and pred != OK:
            self.false_stop += 1
            self.false_stop_unseeable += unseeable
        else:
            self.other += 1


# -- one model --------------------------------------------------------------------


Predictor = Callable[[Path], tuple[list[Detection], float]]


@dataclass
class ModelResult:
    label: str
    weights: str
    classes: list[str]
    scores: list[ClassScore] = field(default_factory=list)
    decisions: Decisions = field(default_factory=Decisions)
    ms: float = float("nan")

    def mean(self, attr: str, over: Iterable[str] | None = None) -> float:
        keep = set(over) if over is not None else None
        vals = [
            getattr(s, attr)
            for s in self.scores
            if s.detectable and s.instances and (keep is None or s.name in keep)
        ]
        vals = [v for v in vals if v == v]
        return float(np.mean(vals)) if vals else float("nan")


def evaluate_model(
    label: str, weights: str, model_classes: list[str], ds: Dataset, cfg: Config, predict: Predictor
) -> ModelResult:
    """Score one model. ``predict(image) -> (detections at a low floor, ms)``.

    One low-threshold prediction per image serves both halves: the detection metrics need
    every box down to conf 0.001, and the decision half filters those to the configured
    per-class floors, which is what the station's detector does with them.
    """
    res = ModelResult(label, weights, model_classes)
    detectable = set(model_classes)
    missing = [c.name for c in cfg.ppe.classes if c.name not in detectable]
    tps, preds, truth, times = [], [], [], []
    for s in ds.samples:
        dets, ms = predict(s.image)
        dets = [d for d in dets if d.name in ds.names]  # classes the dataset does not label
        times.append(ms)
        tps.append(match(dets, s.truth))
        preds.append(dets)
        truth.append(s.truth)
        truth_v = verdict(s.truth, cfg)
        unseeable = bool(missing) and truth_v == OK and verdict(s.truth, cfg, missing) != OK
        res.decisions.add(
            truth_v, verdict(with_floors(dets, cfg), cfg, missing), s.image.name, unseeable
        )
    res.scores = score_classes(ds.names, detectable, tps, preds, truth)
    res.ms = float(np.median(times)) if times else float("nan")
    return res


def ultralytics_predictor(weights: str, imgsz: int) -> tuple[Predictor, list[str]]:
    """Predict through the station's letterbox and model wrapper, at conf 0.001.

    Uses ppe.detector.Detector so the timing and the preprocessing are the station's own,
    with NMS at IoU 0.7 and up to 300 boxes, as for validation.
    """
    import cv2

    from .config import PPECfg
    from .detector import Detector

    det = Detector(
        ModelCfg(
            weights=weights,
            imgsz=imgsz,
            device="cpu",
            half=False,
            conf=0.001,
            iou=0.7,
            max_det=300,
            warmup=True,
            batch=False,
        ),
        PPECfg(classes=[]),
    )
    det.class_ids = []  # every class the model has
    det._floors = {}
    det._conf = 0.001
    names = list(det.names.values())

    def predict(image: Path) -> tuple[list[Detection], float]:
        frame = cv2.imread(str(image))
        t = time.perf_counter()
        dets, _ = det.detect([frame])
        return dets[0], (time.perf_counter() - t) * 1000

    return predict, names


# -- discovery and labels -----------------------------------------------------------


def discover(roots: Iterable[str] = ("models", "model-versions")) -> list[str]:
    """OpenVINO exports first (what stations run), else the .pt; batch-N exports skipped."""
    found = []
    for root in map(Path, roots):
        if not root.is_dir():
            continue
        for pt in sorted(root.glob("*.pt")):
            ir = pt.with_name(pt.stem + "_openvino_model")
            found.append(f"{ir}/" if ir.is_dir() else str(pt))
    return found


def label_for(weights: str) -> str:
    name = Path(weights.rstrip("/")).name
    name = re.sub(r"_openvino_model$", "", re.sub(r"\.pt$", "", name))
    return {"ppe-yolo11s": "v2 (ppe-yolo11s)"}.get(name, name)


# -- output ---------------------------------------------------------------------------


def _f(x: float, nd: int = 3) -> str:
    return "—" if x != x else f"{x:.{nd}f}"


def _pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:.1f} %"


def common_classes(results: list[ModelResult], ds: Dataset) -> list[str]:
    have = set(ds.names)
    for r in results:
        have &= set(r.classes)
    return [
        n
        for n in ds.names
        if n in have and any(s.name == n and s.instances for s in results[0].scores)
    ]


def summary_table(results: list[ModelResult], ds: Dataset, n: str = "5") -> str:
    common = common_classes(results, ds)
    lines = [
        f"**Table {n}.** All model versions on one common test set ({len(ds.samples)} "
        f"image{'s' * (len(ds.samples) != 1)}, "
        f"split `{ds.split}`). mAP over the {len(common)} classes every model detects; decisions "
        f"from the station's own logic at its configured floors, single frame. *Like for "
        f"like* leaves out compliant images that need a class the model lacks (an apron "
        f"worn instead of sleeves).",
        "",
        "| Model | Classes | mAP@0.5 | mAP@0.5:0.95 | Mean recall | False permits "
        "| False stops | False stops, like for like | ms/frame |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        d = r.decisions
        lines.append(
            f"| {r.label} | {len(r.classes)} | {_f(r.mean('ap50', common))} "
            f"| {_f(r.mean('ap', common))} | {_f(r.mean('recall', common))} "
            f"| {_pct(d.false_permit_rate)} ({d.false_permit}/{d.truth_violation}) | "
            f"{_pct(d.false_stop_rate)} ({d.false_stop}/{d.truth_ok}) | "
            f"{_pct(d.false_stop_rate_seeable)} ({d.false_stops_seeable}/{d.truth_ok_seeable}) "
            f"| {_f(r.ms, 1)} |"
        )
    return "\n".join(lines)


def detail_tables(results: list[ModelResult], ds: Dataset) -> str:
    counts = {s.name: s.instances for s in results[0].scores}
    head = "| Class | Instances | " + " | ".join(r.label for r in results) + " |"
    rule = "| --- | --- | " + " | ".join("---" for _ in results) + " |"
    rows_ap50 = [head, rule]
    rows_ap = [head, rule]
    rows_r = [head, rule]
    for n in ds.names:
        by = [next(s for s in r.scores if s.name == n) for r in results]
        rows_ap50.append(f"| {n} | {counts[n]} | " + " | ".join(_f(s.ap50) for s in by) + " |")
        rows_ap.append(f"| {n} | {counts[n]} | " + " | ".join(_f(s.ap) for s in by) + " |")
        rows_r.append(f"| {n} | {counts[n]} | " + " | ".join(_f(s.recall) for s in by) + " |")
    dec = [
        "| Model | Agree | False permits | False stops | of which need a missing class "
        "| Other | Labels: OK / violation / nobody |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        d = r.decisions
        dec.append(
            f"| {r.label} | {d.agree}/{d.images} | {d.false_permit} | {d.false_stop} "
            f"| {d.false_stop_unseeable} | {d.other} | "
            f"{d.truth_ok} / {d.truth_violation} / {d.truth_standby} |"
        )
    return "\n".join(
        [
            "**Table A3.** AP@0.5 per class on the common test set. "
            "— means the model has no such class.",
            "",
            *rows_ap50,
            "",
            "**Table A4.** AP@0.5:0.95 per class on the common test set.",
            "",
            *rows_ap,
            "",
            "**Table A5.** Recall per class at each model's best mean-F1 confidence threshold.",
            "",
            *rows_r,
            "",
            "**Table A6.** Station decisions against the labels, per image "
            "(single frame, configured floors).",
            "",
            *dec,
        ]
    )


def write_outputs(results: list[ModelResult], ds: Dataset, out: Path, meta: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    md = out / "model-evaluation.md"
    md.write_text(
        f"# Model evaluation on a common test set\n\n"
        f"Generated by `python -m ppe.evaluate` on {meta['date']}. {meta['machine']}\n\n"
        + summary_table(results, ds)
        + "\n\n"
        + detail_tables(results, ds)
        + "\n"
    )
    with (out / "model-evaluation.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "model",
                "weights",
                "class",
                "instances",
                "detectable",
                "precision",
                "recall",
                "ap50",
                "ap50_95",
            ]
        )
        for r in results:
            for s in r.scores:
                w.writerow(
                    [
                        r.label,
                        r.weights,
                        s.name,
                        s.instances,
                        s.detectable,
                        s.precision,
                        s.recall,
                        s.ap50,
                        s.ap,
                    ]
                )
    (out / "model-evaluation.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "split": ds.split,
                "images": len(ds.samples),
                "classes": ds.names,
                "models": [
                    {
                        "label": r.label,
                        "weights": r.weights,
                        "classes": r.classes,
                        "ms_per_frame": r.ms,
                        "decisions": {
                            **r.decisions.__dict__,
                            "false_permit_rate": r.decisions.false_permit_rate,
                            "false_stop_rate": r.decisions.false_stop_rate,
                            "false_stop_rate_like_for_like": r.decisions.false_stop_rate_seeable,
                        },
                        "per_class": [s.__dict__ for s in r.scores],
                    }
                    for r in results
                ],
            },
            indent=1,
            default=float,
        )
    )
    return md


TABLE_LABEL = re.compile(r"\*\*Table ([A-Z]?\d+)\.\*\*")


def keep_table_numbers(old: str, new: str) -> str:
    """Give the new block's tables the numbers the old block used, in order, if they match up."""
    labels = TABLE_LABEL.findall(old)
    if len(labels) != len(TABLE_LABEL.findall(new)):
        return new
    it = iter(labels)
    return TABLE_LABEL.sub(lambda _: f"**Table {next(it)}.**", new)


def fill_report(report: Path, blocks: dict[str, str]) -> list[str]:
    """Replace each <!-- evaluation:NAME --> ... <!-- /evaluation:NAME --> block.

    Returns the names filled. Table numbers already in a block are kept, so a report that
    numbers its tables differently (they moved, or others were added before them) stays
    consistent with its own cross-references.
    """
    text = report.read_text(encoding="utf-8")
    done = []
    for name, body in blocks.items():
        start, end = MARK.format(name), END.format(name)
        a, b = text.find(start), text.find(end)
        if a < 0 or b < a:
            log.warning("no %s ... %s block in %s; left it out", start, end, report)
            continue
        body = keep_table_numbers(text[a + len(start) : b], body)
        text = text[: a + len(start)] + "\n" + body.strip() + "\n" + text[b:]
        done.append(name)
    report.write_text(text, encoding="utf-8")
    return done


# -- main -------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data", required=True, type=Path, help="dataset yaml (YOLO format) with the common split"
    )
    p.add_argument("--root", type=Path, help="dataset folder, overriding the yaml's `path`")
    p.add_argument("--split", default="test", help="split to score on (default: test)")
    p.add_argument(
        "-w",
        "--weights",
        nargs="+",
        help="models to compare (default: models/ and model-versions/)",
    )
    p.add_argument(
        "-c", "--config", default="config.yaml", help="station config for the decision rules"
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--limit", type=int, default=0, help="score only the first N images (a quick check)"
    )
    p.add_argument("--out", type=Path, default=Path("docs/report/data"))
    p.add_argument(
        "--report", type=Path, help="also fill the evaluation blocks in this report source"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    cfg = Config.load(args.config)
    ds = load_dataset(args.data, args.root, args.split, args.limit)
    weights = args.weights or discover()
    print(f"{len(ds.samples)} images ({ds.split}), {len(ds.names)} classes, {len(weights)} models")

    results = []
    for w in weights:
        print(f"  {label_for(w):<28}", end="", flush=True)
        try:
            predict, names = ultralytics_predictor(w, args.imgsz)
            r = evaluate_model(label_for(w), w, names, ds, cfg, predict)
        except Exception as exc:  # noqa: BLE001 — report and keep going
            print(f"failed: {exc}")
            continue
        results.append(r)
        print(
            f"mAP@0.5 {_f(r.mean('ap50'))}  false permits {r.decisions.false_permit}  "
            f"false stops {r.decisions.false_stop}  {r.ms:.1f} ms"
        )
    if not results:
        print("no model could be evaluated")
        return 1

    import platform

    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "split": ds.split,
        "images": len(ds.samples),
        "machine": f"CPU: {platform.processor() or platform.machine()}, "
        f"{os.cpu_count()} logical cores.",
    }
    md = write_outputs(results, ds, args.out, meta)
    print(f"\nwrote {md} (+ .csv, .json)")
    print("\n" + summary_table(results, ds))
    if args.report:
        filled = fill_report(
            args.report,
            {"summary": summary_table(results, ds), "detail": detail_tables(results, ds)},
        )
        print(
            f"\nfilled {', '.join(filled) or 'nothing'} in {args.report}; "
            "rebuild with: python docs/report/build_docx.py --pdf"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
