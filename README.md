# Walking Cane Vision — Camera-Based Obstacle Detection

Camera software for an assistive walking cane. The cane's existing Arduino
rig uses ultrasonic and IR sensors, which answer *"is something there?"* but
not *"what is it, and where?"* — this adds a camera that classifies obstacles,
estimates how far away they are, and decides **when it is worth interrupting
the user**.

That last part is the actual engineering problem. A detector running at 15fps
will happily report the same lamp post 15 times a second, and a mobility aid
that buzzes constantly is one the user switches off. Most of the code here
exists to stay quiet.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m src.cli selftest          # whole loop on synthetic frames, no hardware
pytest -q                           # 66 tests, no camera needed
python -m src.cli run --dry-run     # real webcam, prints alerts instead of buzzing
```

`selftest` and `--dry-run` need no Arduino, no camera, and no model file, so
you can see the pipeline work before any hardware is wired up.

## How it decides to speak

```
camera → detector → distance → zone → alert policy → haptics + speech
```

| Stage | What it does |
|-------|--------------|
| **Camera** | `picamera2` on the Pi, OpenCV webcam elsewhere, video file for tuning — all behind one `CameraSource` protocol |
| **Detector** | MobileNet-SSD via onnxruntime; falls back to a model-free motion detector so the repo runs on clone |
| **Distance** | Pinhole model, `Z = f · H / h`, using known real-world heights per class |
| **Zone** | Left / ahead / right, with the centre band widened to 40% because obstacles in your path matter most |
| **Alert policy** | Confirmation, cooldown, and escalation override — see below |
| **Output** | Vibration motors over the existing Arduino serial link, plus optional offline speech |

### The three mechanisms that keep it quiet

1. **Confirmation** — a detection must persist 2 consecutive frames before it
   can alert, which discards single-frame false positives.
2. **Cooldown** — one alert per zone per 1.6s, so a static obstacle is
   announced once as you approach rather than continuously.
3. **Escalation override** — cooldown is *bypassed* when urgency rises, so a
   warning that becomes critical is never silenced by its own earlier alert.
   This is the subtle one: naive cooldown suppresses exactly the alert that
   matters most.

Urgency comes from proximity **and closing speed**. An obstacle 3.5m away is
routine when you are standing still and critical when you are walking into it
at 1.8 m/s, so time-to-contact can escalate a distant obstacle that a pure
distance threshold would ignore.

Haptics fire for every alert; speech is reserved for warnings and above, and
is skipped for unlabelled blobs — *"obstacle somewhere, 2 metres"* is not worth
interrupting someone for.

## Measured behaviour

From the synthetic approach in `src/cli.py selftest` (obstacle closing 4.0m →
0.5m over 2s) and the pipeline test suite:

| Property | Result |
|----------|--------|
| Alerts for an obstacle closing over 10s | **< 20** (vs 150 raw detections) |
| Alerts for a static obstacle held 1s at 15fps | **1** (vs 15 raw detections) |
| Empty scene | **0** alerts, 0 haptic pulses |
| Escalation WARN → CRITICAL inside cooldown | **not suppressed** |
| Calibration round-trip error | **0.00 m** |

The second row is the number that matters: a ~15× reduction in interruptions
with no loss of the critical alert.

## Calibrate before trusting a distance

Every distance depends on one number — focal length in pixels. The default is
a generic 60°-FOV guess and can be off by 30% or more.

```bash
python scripts/calibrate_focal.py --distance 3.0 --height 1.70
python -m src.cli run --focal-px 554.0 --model models/ssd.onnx
```

Stand a known object at a measured distance, drag a box around it, and pass the
printed value back in. Readings taken without this are flagged
`uncalibrated` throughout the code and the CLI warns on startup.

## Arduino protocol

Newline-terminated ASCII, so you can drive the motors from the serial monitor
with no host software running:

```
V<zone>,<intensity>,<duration_ms>\n    pulse one motor (zone 0=left, 1=centre, 2=right)
S\n                                    stop all motors
P\n                                    ping; board replies "OK"
```

## Layout

```
src/
  config.py      typed settings — every threshold lives here
  geometry.py    boxes, zones, pinhole distance, time-to-contact
  camera.py      Pi / webcam / video-file / replay backends
  detector.py    SSD detector + model-free fallback
  alerts.py      confirmation, cooldown, escalation  ← the interesting part
  haptics.py     Arduino serial link
  audio.py       offline TTS / console
  pipeline.py    the capture→alert loop
  cli.py         run / selftest
scripts/
  calibrate_focal.py
tests/           66 tests, all hardware-free
```

Collaborators are injected throughout, so the pipeline runs headless against a
frame list and recording sinks. That is why the alert-policy behaviour is
testable at all.

## Limitations

- **Monocular depth is an estimate, not a measurement.** It assumes a known
  real-world height per class, so an unusually short person or a child reads as
  further away than they are. The ultrasonic sensors remain the authority at
  close range; the camera adds identity and bearing.
- **Unknown objects get an ordering cue, not a distance.** The fallback uses
  where an object meets the ground, which is monotonic but not metric. These
  are marked `distance_is_calibrated=False`.
- **Not validated with users.** The thresholds are reasoned defaults, not
  findings from testing with people who are visually impaired. Anyone taking
  this further should treat `AlertConfig` as the thing to study, not trust.
- **CPU-bound.** MobileNet-SSD on a Pi 4 CPU runs ~5-8fps at 300×300, which is
  adequate for walking pace but not for cycling.

## License

MIT
