# Dual USB Capture

A one-file Python script that shows two USB cameras side by side and writes a
still from **both at once** — triggered by an on-screen button or the
spacebar, after an adjustable timer that starts at 2 seconds.

```
python capture.py
```

## What it does

* **Two live views.** Each camera runs on its own reader thread, so a slow or
  missing camera never freezes the window. The caption under each view shows
  the resolution and the measured frame rate, or `no signal` if the camera
  stops.
* **One shutter, two ways.** The `Capture (Space)` button and the spacebar do
  the same thing. Press either again — or `Esc` — while the timer is running to
  call the shot off.
* **Adjustable timer.** The box beside the button holds the delay in seconds
  (default `2`, range 0–60, arrows step by 0.5). The remaining whole seconds
  count down in large type, with a beep on each tick that the `Beep` checkbox
  silences. Set the timer to `0` to fire the moment you press.
* **Paired files.** Both frames share one timestamped stem, so a pair is
  obvious on disk and nothing is overwritten if you shoot twice in a second:

  ```
  captures/20260901-152626_cam0.jpg
  captures/20260901-152626_cam1.jpg
  ```

  The status line reports how many milliseconds apart the two frames were
  actually read — with ordinary UVC webcams this is usually well under one
  frame time, but the cameras are not genuinely synchronised, so anything
  moving fast will differ slightly between them.

## Install

```
pip install -r requirements.txt
sudo apt install python3-tk        # Debian/Ubuntu only; Tk ships with
                                   # Python on Windows and macOS
```

## Run

```
python capture.py                       # cameras 0 and 1, 2-second timer
python capture.py --list                # which device indices actually open
python capture.py --cameras 0 2         # pick the indices yourself
python capture.py --labels left right   # name them; the names go in the filenames
python capture.py --delay 5             # start the timer at 5 seconds instead
python capture.py --width 1920 --height 1080
python capture.py --format png -o /data/shots
python capture.py --demo                # synthetic cameras, no hardware
```

`--demo` runs the whole UI against two generated test patterns, which is the
quickest way to check the window, the spacebar and the timer on a machine with
nothing plugged in. `python capture.py --help` lists the rest.

### Picking the right cameras

USB camera indices are assigned by the kernel and can move between reboots or
when a device is replugged. Run `--list` to see what opens right now, then pass
the two you want to `--cameras`. If a camera opens but delivers no frames,
it is usually already in use by another program.

## Layout

```
+-------------------------+-------------------------+
| cam0 - 1280x720 - 30fps | cam1 - 1280x720 - 30fps |
|      [ live view ]      |      [ live view ]      |
+-------------------------+-------------------------+
|                    2                              |  <- countdown
+---------------------------------------------------+
| [ Capture (Space) ]  Timer [ 2.0 ] seconds  [x]Beep|
| Saved 20260901-152626_cam0.jpg + ..._cam1.jpg      |  <- status
+---------------------------------------------------+
```

## Tests

```
pip install -r requirements-dev.txt
pytest
```

The tests cover the parts that need neither a camera nor a screen: the timer,
the countdown text, file naming and collisions, frame collection and staleness,
and argument handling.
