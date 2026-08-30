<img src="assets/logo.svg" alt="" width="76" align="right">

# PPE Detection Station

Two camera feeds → YOLOv11s → a Modbus tower light, with a live checklist of the
PPE each class is or is not wearing.

![layout](docs/layout.svg)

- **Two cameras, one inference.** Both frames go into a single batched
  `predict`, so the GPU is launched once per cycle instead of twice.
- **Smooth video.** Capture, inference and drawing run on separate threads. The
  panes repaint at camera rate no matter how slow the model is, and stale frames
  are dropped rather than queued — the feed stays live instead of drifting
  further behind the longer it runs.
- **Boxes that land where the objects are.** Frames are letterboxed (scaled by
  one factor and padded, never squashed) and every box is mapped back through
  the exact inverse. Capture and inference sizes are free to change.
- **An editable checklist.** Required PPE is listed in the UI; each row turns
  green the moment its class is detected. Toggle, add, remove or re-threshold a
  class and it takes effect on the next frame — no restart.
- **Latency you can see.** Every cycle is timed from the camera grab to the
  relay write, broken down per stage, shown in the HUD and appended to CSV.
  `--profile` adds a cProfile hotspot report covering the worker threads.

## Install

```bash
pip install -r requirements.txt
```

The station ships with `models/ppe-yolo11s.pt`, a YOLOv11s fine-tuned on PPE
(150 epochs at 640, `rect=False` — the same square letterbox this code uses at
inference). Point `model.weights` elsewhere to swap it.

## Run

```bash
python main.py                     # the UI, on the cameras in config.yaml
python main.py -c other.yaml       # a different config
python main.py --headless          # no UI; prints the latency breakdown
python main.py --profile           # add a hotspot report on exit
python main.py --headless --seconds 30 --profile   # a fixed benchmark run
```

## Two faces: operator and debug

```bash
python main.py              # operator — what the floor sees
python main.py --debug      # everything above, plus the numbers behind it
```

The header carries the identity and the notice: logo top-left at 76 px with
room around it, and the privacy notice as a full-width banner rather than a
caption. People are filmed at this station all shift; the line that says why,
and that nobody is recording them, is not a footnote and does not get footnote
treatment.

One window, two layouts — deliberately not two windows. A separate debug
window would drift from the operator one, and then debug would no longer be
showing you what production actually does. The mode gates which panels exist,
not which pipeline runs.

| | operator | debug |
| --- | --- | --- |
| Status, checklist, video | yes | yes |
| Checklist rows | read-only, `OK` / `MISSING` / `1/2` | editable, with confidence floors |
| Latency HUD | hidden | shown |
| Decision panel | hidden | shown |
| Annunciator | **speaks** | **muted** |

**Debug mutes the speaker but keeps the timing running.** The decision panel
still counts down to the prompt that would have played. Muting by skipping the
evaluation would hide the behaviour you opened the debug view to look at.

The **decision panel** is what makes hold and confirm times tunable rather than
guesswork — it shows the raw verdict, what is waiting to be confirmed and for
how long, and how much hold each class has left before it stops counting as
seen:

```
raw       violation
candidate violation  (0.14s of 0.40s, in 0.26s)
applied   violation
audio     muted, next would be 2.9s
CLASS HOLD REMAINING
headnet       0/1  hold 0.78/0.80s
Gloves        1/2  hold 0.62/1.00s
```

`ui.mode` in `config.yaml` sets the default; `--debug` and `--operator`
override it per run. The shipped default is `operator` — a station that boots
into debug on the floor is a mistake.

## Slow to appear on launch?

The window and the video feed do not wait on the model. On this machine:
window shown in 60 ms, live video within 113 ms of launch, the model itself
taking ~4 s to load and warm up in the background. If the app used to look
like it was hanging for several seconds before anything appeared, that was
the model load blocking window creation — fixed by moving model loading onto
the pipeline thread instead of the constructor, so a UI can show and start
painting frames immediately.

While the model loads, the banner reads `LOADING MODEL…`, then `MODEL
READY — WAITING FOR FIRST FRAME` if the cameras are the slower part (the
common case on a real industrial PC — see the camera troubleshooting above),
then the real compliance status once the first cycle completes. A model that
fails to load — a bad path, missing weights — reports `MODEL FAILED TO LOAD:
<reason>` instead of leaving the window stuck on "loading" forever.

## No video signal?

Run this first — no threads, no detector, no UI, just OpenCV opening each
configured source directly:

```bash
python -m ppe.camcheck            # probes every camera in config.yaml
python -m ppe.camcheck --scan     # also enumerates cameras this machine sees
python -m ppe.camcheck -v         # + OpenCV's own diagnostic
python -m ppe.camcheck --save frame.jpg   # save what each camera actually sees
```

It works the same on Windows and Linux — `--scan` and the printed causes adapt
to whichever it finds itself running on — and reports exactly where things
stand for each camera: cannot open, opens but never delivers a frame, delivers
a frame that is one flat colour (lens cap, dead sensor), or works. The main
app only ever logs "unavailable, retrying": that message survives on purpose,
so the station keeps trying rather than crashing when a cable comes loose
mid-shift, but it is not built to be a diagnostic.

**Two USB webcams sharing one port's bandwidth is the single most common
reason two cameras "get no signal" together while either works alone** — see
the bandwidth note under `cameras[].fourcc` below before chasing anything
else; it is now the shipped default specifically because of this.

### On Windows

Ranked by how often each one is the actual cause:

0. **Raw video format eating the USB budget — check this first for TWO
   cameras that individually work but not together.** `cameras[].fourcc:
   MJPG` (the shipped default) asks each camera to compress its own frames
   before sending them; without it most UVC webcams — the Logitech C920
   included — hand back raw YUY2, which is **~55 MB/s at 1280x720@30 for one
   camera**, comfortably more than a single USB 2.0 port carries and roughly
   3x over what two share on one hub. That reads as "no signal" or "opens but
   never delivers a frame," not a visible error. `camcheck` (below) prints
   the actual format each camera negotiated and warns if it is not MJPG.
1. **The camera privacy setting, silently.** Settings → Privacy & security →
   Camera → **"Let desktop apps access your camera."** If this is off, a
   Python/OpenCV process is blocked while the built-in Windows Camera app
   keeps working fine — which is what makes this one so easy to chase in the
   wrong direction. Extremely common on a locked-down industrial/kiosk image,
   and it usually fails with no error at all rather than a clear one. Check
   this before anything else.
2. **Another process already has it open.** Most UVC devices allow only one
   reader. Close Teams/Zoom/the Camera app, and check Task Manager for a
   previous crashed run of this program still holding the device.
3. **Wrong backend — try `api: dshow` first.** `api: any` normally resolves to
   MSMF (Media Foundation), but a reference station running these same C920s
   on Windows opens them with `cv2.CAP_DSHOW` explicitly. That is a working
   configuration on the same hardware, so it is the first thing to try rather
   than the third. `--scan` shows which backend answers for which index.
4. **Driver problem.** Device Manager → Cameras (or "Imaging devices") — a
   yellow warning icon means the driver did not install.
5. **USB power management.** Device Manager → the camera's USB Root Hub →
   Power Management → uncheck *"Allow the computer to turn off this device to
   save power."* Looks identical to a camera that "randomly" stops, and is
   common on industrial PCs with aggressive power profiles.
6. **USB bandwidth.** Two cameras at high resolution can exceed one shared
   hub/controller — the second fails to open while the first works alone.
   Move them to separate controllers, or lower `width`/`height`.
7. **Wrong index.** Windows can renumber cameras after a reboot or a replug;
   `--scan` opens indices 0–5 against both backends and reports what answers,
   so this stops being a guess.

### On Linux

1. **Wrong index.** A UVC webcam commonly exposes *two* `/dev/video` nodes —
   one real, one metadata-only — so `source: 1` can be the wrong node of
   camera 0, not camera 1 at all. `--scan` lists every node.
2. **Permissions.** `groups` needs `video`, or root. `sudo usermod -aG video
   $USER`, then log out and back in.
3. **USB bandwidth.** Same as Windows above; split across controllers or
   lower the requested resolution.
4. **Device busy.** `sudo fuser -v /dev/video0`, or reboot.
5. **Backend mismatch.** Set `api: v4l2` explicitly if `api: any` guesses
   wrong.
6. **Containers.** `/dev/video0`/`/dev/video1` need explicit `--device`
   passthrough — not visible by default even in a privileged container.

`-v` prints OpenCV's own diagnostic under any of the above, and it usually
names the cause directly.

## Configure## Configure

Everything lives in `config.yaml`, and the parts you tune most often are also
editable in the UI (the checklist's **Save to config** button writes them back).

| Key | What it does |
| --- | --- |
| `model.weights` | Any YOLOv11 `.pt`, `.onnx` or TensorRT `.engine` |
| `model.imgsz` | Inference size, a multiple of 32. Lower = faster, worse on small objects |
| `model.device` | `auto`, `cpu`, `cuda:0`, `mps` |
| `model.threads` | Inference threads; 0 = every core this process may use |
| `model.batch` | `auto` (batch on GPU, take turns on CPU), or force `true`/`false` |
| `model.half` | fp16 on CUDA — usually ~1.5-2x faster, no measurable accuracy cost |
| `cameras[].source` | Camera index, RTSP/HTTP URL, or a video file |
| `cameras[].width/height` | Requested capture size (see below) |
| `ppe.classes[].required` | Only required classes drive the tower light |
| `ppe.classes[].expect` | `present` (must be worn) or `absent` (detecting it *is* the violation) |
| `cameras[].fourcc` | Compression the camera is asked to use — `MJPG` (default) or `""` for the driver's raw default |
| `ppe.classes[].count` | How many must be on the subject — 2 gloves, 2 sleeves (default 1) |
| `ppe.subject` | Class the checks are applied to; empty checks every detection |
| `ppe.containment` | Fraction of a PPE box that must lie on the subject to count |
| `ppe.classes[].conf` | Per-class confidence floor, overriding `model.conf`; also editable live in the UI |
| `ppe.hold_ms` | How long a class stays "present" after its last sighting |
| `ppe.confirm_sec` | Seconds a status must hold before the lamp follows, per status |
| `ppe.classes[].hold_ms` | Per-class hold override; falls back to `ppe.hold_ms` |
| `tower.*` | Transport, address, and the coil behind each lamp |
| `tower.channels` | Channels on the relay board; all are driven low at connect |
| `tower.reconnect_sec` | Wait between reconnect attempts after a bus failure |
| `audio.file` | Spoken prompt while a violation stands; empty = silent |
| `audio.grace_sec` | Pause before the first prompt — time to comply |
| `audio.repeat_sec` | Gap between repeats while the violation stands |
| `ui.mode` | `operator` (floor) or `debug` (diagnostics, muted audio) |
| `telemetry.csv` | Per-cycle latency log; empty string disables |
| `branding.name` / `.tagline` | Metadata only — not currently shown in the app |
| `branding.logo` | Your mark, shown top-left of the app; `.svg`, `.png` or `.jpg`, kept at its own aspect ratio; relative paths resolve from the config file; empty or missing hides the strip |

### Capture size and accuracy

Capture resolution and `imgsz` are independent, and changing either is safe:
frames are scaled by a single factor and padded to a square, so nothing is
distorted, and `INTER_AREA` is used when shrinking so small objects survive the
downscale. What actually costs accuracy is **effective object size in pixels**:

- Raising capture resolution above `imgsz` buys nothing on its own — the frame
  is scaled down to `imgsz` before inference either way. It does cost bandwidth
  and decode time.
- Lowering `imgsz` is the real trade. 640 is the trained default; 480 or 416 is
  noticeably faster and starts to miss small or distant items. Measure it on
  your own footage rather than guessing.
- If workers are far from the camera, prefer a tighter lens or a higher `imgsz`
  over a higher capture resolution.

`tests/test_detector.py` pins this down: the same scene captured at 0.5x, 0.75x
and 1.5x yields the same detections in the same places, to within 3% of the
frame.

### The shipped model

`models/ppe-yolo11s.pt` detects seven classes, and they are not all the same
kind of thing:

| Class | Configured as | Why |
| --- | --- | --- |
| `headnet` | required | worn PPE |
| `Safetyglasses` | required | worn PPE |
| `Mask` | required | worn PPE |
| `Gloves` | required, `count: 2` | two hands |
| `sleeves` | required, `count: 2` | two arms, correct sleeve type |
| `Wrong Sleeve` | required, `expect: absent` | **a violation class** — see below |
| `person` | `subject` | who the checks are applied to |

**`Wrong Sleeve` is inverted.** Detecting it is the fault, not the pass. Listed
as ordinary required PPE it would turn the tower **green** on exactly the
condition it exists to catch, so it is configured `expect: absent`: the class
gates the light, but compliance means *not* seeing it. The UI marks such rows
with `⊘`, and green keeps its usual meaning — "as the site rules want it" —
so an inverted row reads green while absent and red the moment it appears.

**Two gloves, not "gloves".** A worker has two hands and two arms, so
`Gloves` and `sleeves` carry `count: 2`. A class is only compliant once that
many are on the subject; one glove reads `VIOLATION` and the banner says
`Gloves (1 of 2)` rather than a bare label. The checklist marks these rows
`×2` and shows the tally — `1/2` — instead of a confidence score, because the
tally is the fault.

Counts are the **best single camera's view, never the sum.** Two cameras
watching one worker both see the same two gloves; adding them up would report
four and pass a one-gloved worker on a doubled count. So the front camera
seeing both gloves gives 2, and a side view that can only see one does not veto
it. The hold window applies to the count as well, so a glove the model loses
for a frame does not read as a bare hand.

**`person` is the subject.** PPE only means anything relative to someone
wearing it, so `ppe.subject: person` gates the whole check:

- **Nobody in view → `STANDBY`.** An empty cell has no PPE in it; reporting
  every item as missing would be a nuisance alarm, and nuisance alarms get
  systems ignored.
- **Someone in view → only their equipment counts.** A detection is credited
  to the subject when at least `ppe.containment` of its box (0.5 by default)
  lies inside the subject's. A helmet on a bystander at the back of the cell
  does not dress the person at the door.
- **The largest person is the subject** — the one nearest the camera. Each
  camera picks its own: with two views of one cell, a single global "largest"
  would silently discard whatever the other camera could see.

Off-subject detections are drawn faint in the video panes rather than dropped,
so it stays obvious that the model saw them and the rules set them aside, and
the chosen person is ringed `SUBJECT`. During standby the checklist goes
neutral grey — a screen full of red items under a `STANDBY` banner would read
as violations nobody committed.

**This is fail-open, by construction.** If the model misses the person, the
station stands by instead of alarming. That is the trade you accept for
silencing empty-cell alarms; `ppe.hold_ms` keeps the subject alive across
dropped frames, and setting `ppe.subject:` empty restores checking everything,
everywhere, all the time.

### Choosing a different model

Point `model.weights` at any YOLOv11 `.pt`, `.onnx` or `.engine` and list its
class names in `ppe.classes`. Any name the model does not have is flagged amber
in the UI and forces the tower to `DEGRADED` rather than silently passing — a
class that can never be detected must never read as compliant.

## Audio

A lamp only works on someone facing it. Point `audio.file` at an .mp3 or .wav
and the station speaks while a violation stands:

```yaml
audio:
  file: assets/please_wear_ppe.mp3
  grace_sec: 3.0     # time to comply before the first prompt
  repeat_sec: 3.0    # gap between repeats while it stands
```

`grace_sec` is the part worth keeping. Prompting the instant PPE is missing
trains people to tune it out; a few seconds to finish putting a glove on means
the prompt only ever fires at someone who actually needed telling. Leave `file`
empty for a silent station — nothing else changes.

## The tower light

| Status | Lamp | Meaning |
| --- | --- | --- |
| `OK` | green | The subject is wearing everything required |
| `VIOLATION` | red | A required class is missing, or a forbidden one appeared |
| `STANDBY` | dark | No subject in view — nothing to judge |
| `DEGRADED` | amber | No usable video, or a required class the model lacks |

Detections are unioned across cameras: an item seen by either camera counts as
present. That is what you want for two views of one cell — a front camera sees
the mask and safety glasses, a side camera sees the gloves and sleeves, and
between them they see the whole worker. Each camera selects its own subject
first, so the union is over what both views saw *on the person each is looking
at*.

With one worker in the cell both cameras lock onto the same person and this is
exactly right. With two people in view the cameras can pick different subjects
and the union would blend them — resolving that needs cross-camera identity,
which this does not attempt.

Standby leaves the tower dark. Amber stays reserved for `DEGRADED`, which is a
fault and needs to look like one; an unlit tower cannot be mistaken for a
compliance verdict.

Two guards keep the relay from chattering: `hold_ms` bridges a class that
flickers out for a frame or two, and `confirm_sec` requires a status to stand
for a set time before the lamp follows it.

Both are timed in **seconds, not frames**. Inference rate moves with CPU load,
so a frame count is a different amount of real time from one minute to the
next — the same setting would debounce for 0.1 s under light load and 1.5 s
under heavy load.

`confirm_sec` is asymmetric on purpose:

```yaml
confirm_sec:
  ok: 1.0             # slow to go green — a safety claim
  violation: 0.4      # quick to go red — an alarm
  standby: 1.0        # slow to conclude nobody is there
  degraded: 0.5
```

Going green asserts that a worker is protected; going red asserts that they
might not be. Those are not equally weighty claims and should not take equally
long. `standby: 1.0` matters for the same reason: slow to conclude the cell is
empty, so a blinked `person` detection cannot quietly drop the alarm.

Hold windows are per class, because detection stability is not uniform —
`sleeves` are large and viewpoint-dependent (1.2 s), a `Mask` is reliably seen
head-on (0.7 s), and `Wrong Sleeve` is short (0.4 s) so a violation clears
promptly once the worker fixes it. Only coils that actually changed are written,
and a bus that drops out is retried in the background without stalling
detection — the UI says `tower offline` while that lasts.

**Relay channel numbering.** These boards label their channels from 1 while
Modbus addresses coils from 0, so **board channel N is coil N−1**. A reference
station on this hardware wires green to channel 1 (coil 0) and red to channel 3
(coil 2) — which is what `tower.coils` ships with. That station has no amber or
buzzer at all, so if yours is wired the same way, check what `DEGRADED` and the
buzzer actually reach before relying on them.

At connect the station drives every channel on the board low, not just the ones
it manages, so a coil left energised by a run that crashed cannot survive into
this one.

Wiring is `tower.coils`: a Modbus coil address per lamp. Set
`tower.transport: rtu` with `serial_port`/`baudrate` for a serial relay, or
`tcp` with `host`/`port` for an Ethernet one. `tower.enabled: false` runs the
whole app with no bus at all, which is how the tests and a dev laptop run.

## Running on a CPU

The station is set up for CPU inference out of the box. Four things get it
there, measured on a 4-core Xeon with two 30 fps cameras:

| | end-to-end p50 | inference p50 | cycles in 25 s |
| --- | --- | --- | --- |
| PyTorch `.pt`, batched, untuned | 488 ms | 476 ms | 56 |
| **shipped defaults** | **48 ms** | **28 ms** | **793** |

**OpenVINO instead of PyTorch — 5.2x.** Same weights, same detections: on the
reference image every box lands within 1.2 px and 0.012 confidence of the
PyTorch output. `models/ppe-yolo11s_openvino_model/` is committed; regenerate it
for a different `imgsz` with `python -m ppe.export`.

**One camera per cycle, not two.** A batch of two costs about 2.3x a batch of
one on a CPU *and* makes each frame wait for the other's result. Serving the
cameras in turn halves end-to-end latency at the same per-camera update rate —
and the ONNX/OpenVINO exports are fixed at batch 1 regardless. `model.batch:
auto` batches on a GPU and takes turns on a CPU; set `true`/`false` to force it.

**Threads that don't fight.** OpenCV defaults to a thread per core *inside each
camera thread*, so on 4 cores the decoders and the model were fighting over the
same hardware — inference measured 3x slower inside the app than on its own.
Capture is now pinned to one thread each and the model gets the cores
(`model.threads`, 0 = all). That alone closed the gap: in-app inference now
matches its isolated benchmark exactly.

**Capture paced to `cameras[].fps`.** A live camera paces itself inside
`read()`, but a file or a free-running source will decode flat out and burn a
core producing frames nobody consumes.

### Measuring it yourself

The right backend depends on your CPU, so measure rather than assume:

```bash
python -m ppe.bench                        # every model in models/
python -m ppe.bench --image shift.jpg      # on a real frame, with detection counts
python -m ppe.bench --imgsz 640 512 448    # what a smaller input buys
python -m ppe.export --format onnx         # if OpenVINO is not an option
```

`bench` prints the detection count beside each timing, so a backend that is fast
because it stopped finding things is obvious.

### If it is still too slow

In the order I would try them:

1. **Lower `model.imgsz`.** 640 → 512 → 448 costs roughly 25% each step. Export
   at the size you intend to run (`python -m ppe.export --imgsz 512`), then
   check recall on your own footage — small or distant PPE goes first.
2. **`python -m ppe.export --int8 --data your-data.yaml`.** Usually ~2x again,
   and this CPU has AMX-INT8. It needs *your* dataset yaml: calibrating on
   someone else's images is how quantisation quietly loses recall. Validate
   before trusting it — I have not, since the calibration set is yours.
3. **Raise `ppe.confirm_sec`** rather than chasing frame rate. The tower does
   not need to react in 50 ms; it needs to be right.

## Latency and profiling

Latency is measured **from the camera grab**, not from the start of inference,
so it is the number that matters — how stale the lamp is relative to the world.

```
stage          p50 ms   p95 ms   max ms   share      n
wait             3.95     6.83     7.72    0.9%     58
preprocess       1.43     2.11     3.38    0.3%     58
inference      414.46   443.78   483.71   98.6%     58
postprocess      0.26     0.50     0.50    0.1%     58
logic            0.03     0.06     0.20    0.0%     58
relay            0.00     0.01     0.04    0.0%     58
------------------------------------------------------
end-to-end     417.75   450.92   491.10  100.0%     58
```

| Stage | Covers |
| --- | --- |
| `wait` | Grab → the pipeline picking the frame up |
| `preprocess` | Letterbox and batch |
| `inference` | The forward pass and NMS |
| `postprocess` | Boxes back to source pixels, per-class thresholds |
| `logic` | Compliance state machine |
| `relay` | The Modbus write |
| `render` | UI repaint (main thread, shown in the HUD) |

The HUD shows this live, `telemetry.csv` records every cycle for offline
analysis, and `telemetry.print_every` prints it to the console on an interval.

`--profile` writes `logs/profile.prof` and prints the top functions by self time
and by cumulative time. cProfile is per-thread, so the pipeline thread opts
itself in — without that, the profile would show only an idle main thread. Dig
in further with:

```bash
python -m pstats logs/profile.prof
pip install snakeviz && snakeviz logs/profile.prof
```

(The run above is CPU-only inference on a container — `inference` at 98.6% is
exactly what you would expect. On a GPU it drops to a few milliseconds and the
other stages start to matter.)

## Layout

```
main.py              CLI entry point
config.yaml          all configuration
ppe/
  config.py          typed config, load/save/validate
  capture.py         one thread per camera, latest frame wins
  letterbox.py       aspect-preserving resize + the exact box inverse
  detector.py        YOLOv11, both cameras in one batched call
  tower.py           compliance state machine + Modbus tower light
  pipeline.py        the loop tying it together
  latency.py         per-stage timing, CSV, thread-aware profiler
  runtime.py         CPU thread budget: capture vs inference
  annunciator.py     the spoken prompt, and when not to give it
  subject.py         who is being checked, and whose gear counts
  export.py          `python -m ppe.export` — OpenVINO / ONNX conversion
  bench.py           `python -m ppe.bench` — measure backends on your machine
  camcheck.py         `python -m ppe.camcheck` — diagnose "no video signal"
  ui.py              Qt: video panes, editable checklist, latency HUD
models/              the fine-tuned PPE weights
assets/logo.svg      placeholder personal mark — swap for your own
docs/layout.svg      architecture diagram
tests/               319 tests
```

## About

<!--
  Replace the placeholder text below with your own background, and swap
  `assets/logo.svg` for your logo. The app only shows the logo mark itself
  (top-left, at its own aspect ratio) — `branding.name` and `branding.tagline`
  in config.yaml are metadata, not currently rendered anywhere in the UI.
-->

<img src="assets/logo.svg" alt="" width="60" align="left" hspace="14" vspace="4">

**Your Name** — Computer Vision & Industrial Automation

_Placeholder — replace with a short paragraph on your background: the domains
you build in, the kind of systems you have shipped, and what led to this
project._

<br clear="left">

| | |
| --- | --- |
| **Focus** | _e.g. real-time vision, industrial automation, edge deployment_ |
| **Built with** | Python · PyTorch · Ultralytics YOLO · OpenCV · Qt · Modbus |
| **Links** | _website · GitHub · LinkedIn_ |

The logo above is a placeholder mark. To use your own, drop the file in
`assets/` and point `branding.logo` at it — nothing in the code needs to
change, and an unreadable or missing file is ignored rather than fatal.

## Tests

```bash
pip install pytest ruff
pytest                       # 319 tests
ruff check ppe main.py tests
```

The suite runs without a camera, a GPU, a display or a Modbus device: cameras
are synthetic video files, the UI runs on Qt's offscreen platform, and the bus
is a fake that records coil writes. `tests/test_detector.py` does use the real
model — including a check that our box inverse agrees with Ultralytics' own
`scale_boxes` to within half a pixel — and skips itself if the weights are not
available.
