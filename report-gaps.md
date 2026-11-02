# Report gaps: what `internship-report.md` still needs

This is the working list that goes with `internship-report.md`. It is in four parts, as the
brief asked:

1. placeholders to fill;
2. discrepancies between the project record (Part B of the brief) and the code;
3. figures to produce;
4. Part D items the repository could not answer.

Section numbers refer to the report.

**Read this first.** Three items are not editorial. Settle them before the report is
submitted:

- **Training data provenance and authorship** (2.9 and 2.10 below). The training checkpoint
  names a cloud dataset platform's export, and a Windows user profile belonging to another
  individual. Either can contradict statements in the report.
- **Declaration of AI assistance** (1.1 below). The repository's commit history records
  AI-assisted development, and this report was drafted with AI assistance. Check what the chair
  and the examination regulations require, and declare it.
- **The emergency stop** (2.1 below). The report says plainly that the e-stop currently acts
  only through the relay module and software. Make sure EHS and your supervisor have seen that
  before the report goes out.

---

## 1. Placeholders left in the report

### 1.1 Front matter

| Placeholder | Where | What is needed |
| --- | --- | --- |
| `[matriculation number]` | Cover | Your matriculation number. Left exactly as the brief required. |
| `[start date]` to `[end date]` | Cover | Internship dates. Left as required. |
| `[name, designation]` | Cover, Acknowledgements | Industrial supervisor. Left as required. |
| `[name, chair]` | Cover, Acknowledgements | Academic supervisor. Left as required. |
| `[name, designation of subject matter expert(s)]` | Acknowledgements | The SMEs you want to thank, or delete the clause. |
| Declaration on the use of AI tools | Cover | The wording the chair or regulations require. The placeholder states the facts: the commit trailers record an AI coding assistant, and the report was drafted with AI assistance. |

### 1.2 Body

| Placeholder | Section | What is needed |
| --- | --- | --- |
| Site description of the Pune facility | 1.2 | Role of the site, its products and processes, your department and reporting line. Verify against approved public material. |
| Industrial PC specification | 8.1 | CPU, cores, RAM, storage, OS, USB ports and generation, network, enclosure rating, from the procurement record. No make or model. |
| SOP summary confirmation | 8.6 | Check the four-item summary against the approved SOP. |
| Gemba walk results | 10.3 | Number of stations surveyed, the main variations found, how stations were grouped for camera placement. |
| Camera vendor evaluation outcome | 10.4 | Shortlist against the specification, the choice and the reasons. No supplier names or prices. |
| Droplet transfer: recommended method and rationale | 11.2 | Not in your source notes. |
| Droplet transfer: literature references | 11.2 | Add to the reference list. |
| Roadmap contribution | 11.4 | What you actually prepared, without disclosing roadmap content. Delete if not permitted. |
| Closing reflection | 13 | Your own paragraph. |
| Dataset composition | 4.7, Table 6 | Image counts per split and instances per class (see 4.1 below). |
| Per-class metrics | 5.4, Table 9 | From a validation run (see 4.2 below). |
| After-optimisation stage breakdown on the station | 9.1 | From the station's `logs/latency.csv`, with the relay enabled. |

### 1.3 References

| Placeholder | What is needed |
| --- | --- |
| [13] EHSRA | Document number, revision and date. |
| [14], [15] MOC records | Record numbers and dates for the initial installation and the relocation. |
| [16] SOP | Document number, revision and date. |
| [4] RF-DETR | Currently cites the software repository. Replace it with the paper if you prefer a paper citation, and verify the bibliographic details. |

### 1.4 Figures

See section 3.

---

## 2. Discrepancies between the project record and the code

These are stated in the report at the point they arise, and collected in Appendix C
(Table 25). Items marked **safety** affect the fail-safe behaviour.

| # | Topic | Project record (Part B) | Code or repository | Report |
| --- | --- | --- | --- | --- |
| 2.1 | **Safety:** emergency stop | "Hard-wired to the machine and independent of the detection system" | NO contact into relay module DI1. `config.yaml:198-206` says no safety relay is fitted and this input is the only thing the e-stop does. You told me in this session that the safety relay work is pending, which agrees with the code. | 3.1, 7.6, 12.2 item 1 |
| 2.2 | Grace timer and gloves | Grace exists so "an operator whose glove has slipped" can correct | Gloves, safety glasses and Wrong Sleeve have `grace_sec: 0`. Only head net, mask and sleeves get the 5 s. | 7.4 |
| 2.3 | Modbus thread | UI and Modbus on their own thread, off the critical path | Modbus reads and writes run on the pipeline thread (`ppe/pipeline.py:282-285`). The UI runs on the Qt main thread. | 6.2 |
| 2.4 | Required PPE | Glasses, gloves, sleeves, mask | The model and config also require a **head net**, and treat **Wrong Sleeve** as forbidden. Confirm the head-net requirement against the station's risk assessment. | 4.1 |
| 2.5 | Compliance-state schema | Open question | Resolved for sleeves by the explicit `Wrong Sleeve` class. **Not** resolved for prescription spectacles or carried gloves. The relational test is only "item box inside the person box" (gloves 0.1, sleeves 0.3, others 0.5). | 4.5 |
| 2.6 | Capture configuration | 640×480 at 30 fps, raw; MJPEG rejected | Shipped `config.yaml:31-47` requests 1280×720 at 30 fps, **MJPG**, on both cameras. Confirm what the station's own `config.yaml` actually runs, and change the default before rollout. | 6.5, 12.2 item 12 |
| 2.7 | Original capture size | 1280×1080 at 10 fps | 1280×720 at 10 fps in `camcheck --modes` (`README.md:138`). 1280×1080 is probably a slip. | 6.5 |
| 2.8 | Scheduler history | Freshest-frame selector replaced by strict alternation | Git history shows: batch both cameras (first commit), then round-robin, then a starvation fix (commit `4396155`) for a camera served alone not spending its turn. No freshest-frame selector appears in this repository's history. It may have existed in earlier code outside it; confirm. | 6.3 |
| 2.9 | Dataset provenance | Site images only; no public datasets; no cloud annotation platforms | The checkpoint's `train_args.data` path reads `...\v4 -roboflow only\ppe-yolo11-roboflow-only\datasets\roboflow\data.yaml`. That may only mean the Roboflow *export format*, but it may mean data came from, or went through, Roboflow. Confirm before submission. The report states the discrepancy without naming the platform. | 4.2 |
| 2.10 | Training authorship | Implied: your work | The same path, and `save_dir`, sit under a Windows user profile with **another person's name** (`C:\Users\<name>\...`). It is also visible in `models/*/metadata.yaml`, which is committed. If a colleague ran the training, say so in the report and acknowledge them. Either way, consider scrubbing the path from the committed metadata. | 5.2, Appendix B.2 (redacted) |
| 2.11 | Rollout scope | 45 machines | Installer and `docs/DEPLOYMENT.md` say **32**. You told me in this session that the number changed from 45 to 32. The report uses 45 as "in scope when specified" and flags 32; update to whichever is current. | 10.1 |
| 2.12 | Debounce change | "Reduced" | Was `confirm_frames: 3` (cycles) plus a global hold of 1.5 s. Now per-status seconds: OK 1.0, violation 0.4, standby 1.0, degraded 0.5 (commit `f8cba42`). Consistent with "reduced" at PyTorch cycle rates. | 7.3 |
| 2.13 | Library versions | — | Trained with Ultralytics 8.4.126, exported with 8.4.130, run with 8.3.253 (pinned below 8.4 on purpose). | 3.3, 12.2 item 13 |
| 2.14 | **Safety:** one frozen camera | — | Its last detections stay in force indefinitely. Verified by simulation. | 7.6, 12.2 item 2 |
| 2.15 | **Safety:** exception in inference | — | The thread dies without blanking the board, so the motor stays enabled. Verified by simulation. | 7.6, 12.2 item 3 |
| 2.16 | **Safety:** module watchdog | — | `docs/DEPLOYMENT.md:190-193` relies on the relay module's communication watchdog. Nothing configures it or records a test. | 7.6, 12.2 item 4 |
| 2.17 | DO5 = coil 4 | — | `config.yaml:140-145` asks for this to be checked against the module manual. The project record does not say it was. | 7.5 |

### 2.1 Content from this session that is not in Part B

Confirm or delete each of these.

- **Camera specification** (10.4): USB 3 direct to the PC, 2 m cable, working distances front
  0.5–1.0 m and side 1.0–1.5 m. From our camera-vendor correspondence.
- **FAT and SAT protocols with pulled-wire tests** (10.2, 8.2). Prepared in this session.
- **Integrated tower redesign and the 53 % volume reduction** (10.5): 1.63 m³ to 0.76 m³, taking
  the tower as the grinder's own envelope. From your slide notes.
- **Link between the panel-size comment and the integrated design** (8.7). An inference; confirm.
- **Rollout dates** from the slide (10 machines by 20 Nov, 20 by 10 Dec). Not included. Add them
  to Chapter 10 if wanted; note that 10 + 20 = 30, not 32 or 45.

---

## 3. Figures to produce or source

### 3.1 Produced for the report

These are generated from repository data and stored in `docs/report/figures/`.

| Fig. | File | Source |
| --- | --- | --- |
| 1 | `fig-architecture.svg` | Drawn from `config.yaml` and the code. No addresses. |
| 4 | `fig-training-losses.png` | Checkpoint `train_results`, reproduced in `docs/report/data/training-results.csv` |
| 5 | `fig-training-metrics.png` | Same |
| 7 | `fig-threads.svg` | Drawn from `ppe/pipeline.py`, `ppe/capture.py` and `ppe/ui.py` |
| 8 | `fig-decision-flow.svg` | Drawn from `ppe/subject.py` and `ppe/tower.py` |
| 9 | `fig-hold-windows.png` | Schematic using `config.yaml` values for Gloves |
| 10 | `fig-stop-times.png` | Measurements in commit `0d12057` (simulated against the shipped config) |
| 12 | `fig-latency-before-after.png` | `README.md:637-645` (development host) |

The SVGs render on GitHub. If your final format is PDF or Word, convert them, for example by
opening each in a browser and printing to PDF, or with `rsvg-convert`.

### 3.2 Still required

These are placeholders in the report. None can contain a part, a drawing or an identifiable
face.

| Fig. | What it should show | Where to get it |
| --- | --- | --- |
| 2 | Operator UI in operator mode: status headline, checklist, both camera views | Screenshot at the station with nobody in frame, or cropped or blurred |
| 3 | Annotation workflow: capture → local pre-labels → correction → split → train → export → deploy → capture | Draw it, or ask me to |
| 6 | Normalised confusion matrix, 7 classes plus background | `confusion_matrix_normalized.png` from the validation run (see 4.2 below) |
| 11 | Installed station: panel, tower light, camera stands, monitor | The 3D model requested from design is safest; otherwise a photo with machine and people excluded |
| 13 | Three panels of correct operation: compliant front view; occluded glove credited; bystander set aside | Capture-mode frames, faces blurred |
| 14 | Three panels of failure cases: side-view miss; glove–sleeve confusion; angled posture | Capture-mode `miss` and `violation` frames, faces blurred |

### 3.3 Existing image assets in the repository (D.12)

| File | What it is | Usable? |
| --- | --- | --- |
| `assets/logo.svg` | Placeholder branding mark | No |
| `assets/thumbs_up.svg`, `assets/thumbs_down.svg` | UI glyphs for PPE present or missing | No |
| `docs/layout.svg` | Early pipeline diagram | **No**: it says "one batched call / one GPU launch", which is no longer true. Replaced by Figure 7. |

There are no detection examples, confusion matrices, training plots, UI screenshots or
installation photos in the repository.

---

## 4. Part D items that could not be determined from the repository

| Item | What is missing | How to get it |
| --- | --- | --- |
| **D.2** Dataset composition | Image counts, split sizes, per-class instance counts. The dataset and its `data.yaml` are not in the repository. | On the training machine: count the files in `images/` and `labels/` per split, and count class IDs in the label files. |
| **D.3** Optimiser actually used | `optimizer: auto`; the resolved choice is not stored | The training log's first lines, or `args.yaml` in the run directory |
| **D.3** GPU model | Only "CUDA device 0" is recorded | The training machine |
| **D.4** Per-class P/R/mAP, confusion matrix | Not stored in the checkpoint | `yolo detect val model=models/ppe-yolo11s.pt data=<data.yaml> split=val plots=True`, then archive the run directory with the model |
| **D.4** Other model variants | Only YOLO11s is in the repository | Nano and RF-DETR run directories, if kept |
| **D.5** Nano export precision | No nano export in the repository. The deployed *s* export **is FP32**: all 182 float constants are `f32`, `quantize: null`, static shape, batch 1. | Re-export nano with `python -m ppe.export -w yolo11n-trained.pt` and time both with `python -m ppe.bench` |
| **D.6** Prior debounce in seconds | History records only `confirm_frames: 3` | Not recoverable. It depended on the cycle rate at the time. |
| **D.7** Relay module watchdog | Whether it is enabled, and its timeout | Module web page or configuration. Test by pulling the PC's network cable with the motor running. |
| **D.8** Station's actual capture configuration | The repository shows only the default | `C:\ProgramData\PPE Station\config.yaml` on the station PC (if installed), or the station's working copy |
| **D.10** Station stage breakdown, relay enabled | Only development-host and container figures are in the repository | `logs/latency.csv` on the station |
| **D.11** Exact Python version on the station | CI builds with 3.11. If the station runs from source, its own interpreter applies. | `python --version` on the station, or `PPEStation.exe --version` |
| `Wrong Sleeve` definition | The annotation guideline for this class is not recorded. The config comment says "a sleeve that can be caught". | Your annotation notes. Add one sentence to Section 4.1. |

### 4.1 Values the report does give

These are the D items determined from the code. Each is cited by file and line in the report.

- **D.1:** class schema and roles (Table 4); compliance function (Appendix A.1).
- **D.3:** training hyperparameters (Table 7).
- **D.4:** all-class metrics and training curves (Table 8; Figures 4 and 5).
- **D.5:** deployed export settings (Table 14).
- **D.6:** every timer and threshold (Tables 15 and 16).
- **D.7:** failure behaviour (Table 19).
- **D.8:** threads and scheduling (Table 11, Section 6.3).
- **D.9:** Modbus map (Table 18).
- **D.10:** latency instrumentation (Tables 20 and 21).
- **D.11:** environment (Table 3).
