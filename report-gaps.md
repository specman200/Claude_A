# Report gaps: what the internship report still needs

These notes go with the RWTH version of the report:

- **Source:** `internship-report.md`.
- **Word file:** `docs/report/internship-report.docx`, built with
  `python docs/report/build_docx.py --pdf`.
- **Full draft:** `docs/report/internship-report-full-draft.md`, the long version, kept as the
  reference for the attachments and for any section you want to expand.

**Format and length (RWTH, SPO 2021).** The report has:

- 12 pt Times New Roman, 1.5 line spacing, A4 with 2.5 cm margins;
- page numbers in the footer, with none on the cover;
- a table of contents with page numbers.

Tables, captions and code are set smaller and single-spaced, as is usual. The body, from
Introduction to Conclusion, is **24 pages** (last page half full), against the guideline of about 24, excluding the
cover, attachments and literature. Filling the placeholders below, the supervision reflection in
particular, will bring it to about 24. If you add more than about a page, shorten Chapter 2 or
Section 5.3 first.

**Read this first.** Three items are not editorial. Settle them before the report is
submitted:

- **Training data provenance and authorship** (2.9 and 2.10 below). The training checkpoint
  names a cloud dataset platform's export, and a Windows user profile belonging to another
  individual. Either can contradict statements in the report.
- **Declaration of AI assistance.** The repository's commit history records AI-assisted
  development, and this report was drafted with AI assistance. Check what the chair and the
  examination regulations require. The RWTH guideline page you sent does not mention it; add the
  declaration after the cover if it is required.
- **The emergency stop** (2.1 below). The report says plainly that the e-stop currently acts only
  through the relay module and software. Make sure EHS and your supervisor have seen that before
  the report goes out.

---

## 0. Station configuration (received 2 November 2026)

The report now describes the station's own `config.yaml`. These changes from the repository's
default affect the safety argument and need an EHS position before rollout.

| Setting | First configuration (repository) | Station | Effect, simulated with the station's own logic |
| --- | --- | --- | --- |
| Gloves count | 2 | **1** | A worker with **one bare hand is compliant indefinitely**. Before, removing one glove stopped a running machine after 5.5 s. |
| Sleeves count | 2 | **1** | The same, for one bare forearm |
| Correction window: gloves, glasses | 0 s | **5 s** | Motor stops 6.5–6.6 s after the item leaves view (was 1.5–1.6 s) |
| Correction window: default (head net, mask, sleeves, apron) | 5 s | **8 s** | Stops after 9.2–9.7 s (was 6.2–6.7 s) |
| Wrong Sleeve | hold 0.4 s, window 0 s | hold 1.2 s, window 0 s | Still stops in 0.5 s once it appears |
| Person confidence floor | 0.25 | 0.70 | Fewer background people; the operator is lost more easily, which means STANDBY and the motor stopping |
| Occlusion window | 5 s | 8 s | Has no effect with a count of 1 |
| Model | `ppe-yolo11s_openvino_model` | `best-apron_openvino_model` | **Not in the repository.** Class list, training record and metrics are needed for Attachment A. |
| Cameras | 1280×720, `api: any` | 640×480 and 640×360, DirectShow | Matches the validated low-resolution setting |

Three things to know about the station file itself:

- **The comments no longer match the values.** The comment above the classes still says that
  gloves, glasses and Wrong Sleeve get 0 s. Correct it, so the next person reading the file is not
  misled.
- **It references files the repository does not have:** `models/best-apron_openvino_model/`,
  `assets/please_wear_ppe.mp3` and `assets/logo.png`. If it replaced the repository's
  `config.yaml` as it stands, the installer's self-test and the CI build would fail. Add those
  files first, or keep the station file local.
- **It contains the station's relay address.** That address is kept out of the report.

The full draft (`docs/report/internship-report-full-draft.md`) still describes the first
configuration.

---

## 0a. Model versions (checkpoints received 2 November 2026)

Table 4 and Figure A2 of the report compare the six checkpoints. The data is in
`docs/report/data/model-versions.csv`.

- **`ppe-yolo11s.pt` (repository) and `best-yolov11s_v2.pt` are the same model.** Their weights are
  bit-identical; v2 is a seeded re-run dated 3 September. v1 and v3 were not supplied. If they
  existed, say so in Section 4.4.
- **v4, v5 and v6** have identical settings and differ only in their data. Their epoch times grow
  from about 60 s to 94 s on the same GPU. Their metric gains (mAP@0.5:0.95 from 0.840 to 0.844;
  v6 recall 0.964) are within what separate validation splits can explain.
- **`nanobest_3.pt`** was trained on a cloud notebook against a different dataset copy, with
  different settings and Ultralytics 8.4.135. It scores 0.78 mAP@0.5:0.95 after one epoch, which is
  implausible for a smaller model on the same data. **Check its training and validation split for
  overlap** before citing its scores.
- **`best-apron.pt`** has 8 classes in a *different index order*: person, headnet, Gloves, Mask,
  Safetyglasses, sleeves, Wrong Sleeve, Apron. The station matches classes by name, so the order
  is harmless there, but any tool that uses class indices will be wrong. It was trained on a
  "combined" dataset, fine-tuned from an earlier run, under **another user's Windows profile,
  named by an employee number**. Add that to the authorship question in 2.10, and keep the number
  out of anything you publish. It is not in the report or the CSV.
- **The nano anomaly is resolved.** The `Export and compare models` workflow converted every
  checkpoint with `ppe.export` (FP32) and timed it with `ppe.bench` on one 4-core CI CPU. Nano ran
  at 16.5 ms against 34–35 ms for the small models in OpenVINO, and 40 ms against 122 ms in
  PyTorch. The raw output is `docs/report/data/model-benchmark.md`. Confirm on the station PC if
  you want the station's own numbers.
- **`best-apron` is now the repository default** (`models/best-apron_openvino_model/`). The
  earlier versions' OpenVINO exports are in `model-versions/`. The installer does not ship that
  folder.
- **Personal paths.** The export metadata no longer names the trainers' user folders. The `.pt`
  checkpoints still record them inside their training arguments, and the installer ships
  `models/best-apron.pt` and `models/ppe-yolo11s.pt`. If that matters, re-save the checkpoints
  without the training arguments, or drop the `.pt` files from the installer; the station only
  needs the OpenVINO folder.
- **The one comparison that would decide between versions is still missing:** all six evaluated on
  one fixed test set of station footage, per class (open item 6 in the report).

---

## 0b. Table 5: common-dataset comparison (run it yourself)

Run this on the PC that holds the dataset. The images never leave it.

```bash
python -m ppe.evaluate --data data.yaml --root "<folder that contains images/ and labels/>" \
       --report internship-report.md
python docs/report/build_docx.py --pdf
```

This fills Table 5 (Section 4.4) and Tables A3–A6 (Attachment A). Then write the one-paragraph
interpretation that follows Table 5. `--limit 50` gives a quick check first.

Two things to know:

- **`data.yaml` names a OneDrive path with an employee number.** The tool doesn't need it,
  because `--root` overrides it. Consider removing the `path:` line from the committed file.
- **The test split must be held out from every model's training.** best-apron was trained on
  this combined dataset; if its `test` images were not held out, its scores here are inflated.
  Confirm which images were used for training.

---

## 1. Placeholders left in the report

### 1.1 Cover page (RWTH requirements)

| Placeholder | What is needed |
| --- | --- |
| `[matriculation number]` | Your student ID |
| `[company address, Pune, India]` | The company's address, as RWTH requires on the cover |
| `[start date]` to `[end date]` | Internship dates |
| `[name, designation]` | Industrial supervisor (trainer) |
| `[name, chair]` | Academic supervisor |
| Stamp and signature box | **Company stamp and the trainer's signature**, on a printed copy or a scan |

### 1.2 Body

| Placeholder | Section | What is needed |
| --- | --- | --- |
| Site description of the Pune facility | 1.2 | Role of the site, products and processes, your department |
| **Supervision reflection** | 9.1 | Required by the guidelines: how supervision worked, the independence you had, support from EHS, maintenance and IT, what worked and what didn't. About 200–300 words. |
| Industrial PC specification | 7.1 | CPU, RAM, storage, OS, USB generation, enclosure rating. No make or model. |
| Droplet transfer: recommended method | 9.2 | Not in your source notes |
| Dataset composition and per-class metrics | Attachment A | See 4.1 and 4.2 below |
| References [13]–[16] | References | Document and record numbers for the EHSRA, both MOCs and the SOP |

### 1.3 Figures still required (Attachment F)

The required figures are:

- operator interface screenshot;
- the installed station (the 3D model is safest);
- examples of correct operation and of failures, with faces blurred;
- the confusion matrix;
- the annotation workflow.

None may show a component or an identifiable face.

---

## 2. Discrepancies between the project record and the code

These are stated in the report where they arise, and collected in Attachment D (Table D2). Items marked **safety** affect the fail-safe behaviour.

| # | Topic | Project record (Part B) | Code or repository | Report |
| --- | --- | --- | --- | --- |
| 2.1 | **Safety:** emergency stop | "Hard-wired to the machine and independent of the detection system" | NO contact into relay module DI1. `config.yaml:198-206` says no safety relay is fitted and this input is the only thing the e-stop does. You told me in this session that the safety relay work is pending, which agrees with the code. | 3, 6.4, 11 |
| 2.2 | Grace timer and gloves (**now resolved by the station config**: gloves get 5 s) | Grace exists so "an operator whose glove has slipped" can correct | Gloves, safety glasses and Wrong Sleeve have `grace_sec: 0`. Only head net, mask and sleeves get the 5 s. | 6.3 |
| 2.3 | Modbus thread | UI and Modbus on their own thread, off the critical path | Modbus reads and writes run on the pipeline thread (`ppe/pipeline.py:282-285`). The UI runs on the Qt main thread. | 5.1 |
| 2.4 | Required PPE | Glasses, gloves, sleeves, mask | The model and config also require a **head net**, and treat **Wrong Sleeve** as forbidden. Confirm the head-net requirement against the station's risk assessment. | 4.1 |
| 2.5 | Compliance-state schema | Open question | Resolved for sleeves by the explicit `Wrong Sleeve` class. **Not** resolved for prescription spectacles or carried gloves. The relational test is only "item box inside the person box" (gloves 0.1, sleeves 0.3, others 0.5). | 4.2 |
| 2.6 | Capture configuration | 640×480 at 30 fps, raw; MJPEG rejected | Shipped `config.yaml:31-47` requests 1280×720 at 30 fps, **MJPG**, on both cameras. Confirm what the station's own `config.yaml` actually runs, and change the default before rollout. | 5.3 |
| 2.7 | Original capture size | 1280×1080 at 10 fps | 1280×720 at 10 fps in `camcheck --modes` (`README.md:138`). 1280×1080 is probably a slip. | 5.3 |
| 2.8 | Scheduler history | Freshest-frame selector replaced by strict alternation | Git history shows: batch both cameras (first commit), then round-robin, then a starvation fix (commit `4396155`) for a camera served alone not spending its turn. No freshest-frame selector appears in this repository's history. It may have existed in earlier code outside it; confirm. | 5.2 |
| 2.9 | Dataset provenance | Site images only; no public datasets; no cloud annotation platforms | The checkpoint's `train_args.data` path reads `...\v4 -roboflow only\ppe-yolo11-roboflow-only\datasets\roboflow\data.yaml`. That may only mean the Roboflow *export format*, but it may mean data came from, or went through, Roboflow. Confirm before submission. The report states the discrepancy without naming the platform. | 4.1 |
| 2.10 | Training authorship | Implied: your work | The same path, and `save_dir`, sit under a Windows user profile with **another person's name** (`C:\Users\<name>\...`). It is also visible in `models/*/metadata.yaml`, which is committed. If a colleague ran the training, say so in the report and acknowledge them. Either way, consider scrubbing the path from the committed metadata. | 4.3 (training described without naming who ran it) |
| 2.11 | Rollout scope | 45 machines | Installer and `docs/DEPLOYMENT.md` say **32**. You told me in this session that the number changed from 45 to 32. The report uses 45 as "in scope when specified" and flags 32; update to whichever is current. | 9.1 |
| 2.12 | Debounce change | "Reduced" | Was `confirm_frames: 3` (cycles) plus a global hold of 1.5 s. Now per-status seconds: OK 1.0, violation 0.4, standby 1.0, degraded 0.5 (commit `f8cba42`). Consistent with "reduced" at PyTorch cycle rates. | 6.2 |
| 2.13 | Library versions | — | Trained with Ultralytics 8.4.126, exported with 8.4.130, run with 8.3.253 (pinned below 8.4 on purpose). | Attachment A |
| 2.14 | **Safety:** one frozen camera | — | Its last detections stay in force indefinitely. Verified by simulation. | 6.4 |
| 2.15 | **Safety:** exception in inference | — | The thread dies without blanking the board, so the motor stays enabled. Verified by simulation. | 6.4 |
| 2.16 | **Safety:** module watchdog | — | `docs/DEPLOYMENT.md:190-193` relies on the relay module's communication watchdog. Nothing configures it or records a test. | 6.4 |
| 2.17 | DO5 = coil 4 | — | `config.yaml:140-145` asks for this to be checked against the module manual. The project record does not say it was. | Attachment B |

### 2.1 Content from this session that is not in Part B

Confirm or delete each of these.

- **Camera specification** (9.1): USB 3 direct to the PC, 2 m cable, working distances front
  0.5–1.0 m and side 1.0–1.5 m. From our camera-vendor correspondence.
- **Acceptance test protocols with pulled-wire tests** (9.1). Prepared in this session.
- **Integrated tower redesign and the 53 % volume reduction** (9.1): 1.63 m³ to 0.76 m³, taking
  the tower as the grinder's own envelope. From your slide notes.
- **Link between the panel-size comment and the integrated design** (9.1). An inference; confirm.
- **Rollout dates** from the slide (10 machines by 20 Nov, 20 by 10 Dec). Not included. Add them
  to Section 9.1 if wanted; note that 10 + 20 = 30, not 32 or 45.

---

## 3. Figures

### 3.1 Produced for the report

In the RWTH version, Figures 1–6 are in the body, and the training losses and hold-window
figures are in the attachments.

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

Each SVG has a PNG copy, which the Word build uses.

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

These are the D items determined from the code.

- **In the RWTH version:**
  - training settings and all-class metrics: Attachment A;
  - per-class thresholds and timings, and the Modbus map: Attachment B;
  - failure behaviour: Attachment C;
  - discrepancies: Attachment D.
- **In the full draft** (`docs/report/internship-report-full-draft.md`): every value cited by file
  and line, with the code listings and the configuration in its Appendices A and B.
