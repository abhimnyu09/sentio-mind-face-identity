# 🧠 Sentio Mind — Named Face Identity + Energy Report

> **IIT Mandi × Sentio Mind** · Project 1  
> Automatically identifies students from CCTV footage using reference photos, computes per-person energy scores, and generates a clean HTML report + integration-ready JSON.

---

## 📋 Table of Contents

- [What This Does](#what-this-does)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Setup on macOS](#setup-on-macos)
- [Setup on Windows](#setup-on-windows)
- [Running the Solution](#running-the-solution)
- [Understanding the Output](#understanding-the-output)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)

---

## What This Does

This system takes:
- A folder of **reference photos** (`known_faces/`) — one photo per student, filename = student name
- A **CCTV video file** (`video_sample_1.mov`)

And produces:
- **Named student profiles** with real identity matched from reference photos
- **Energy scores** per person computed from brightness, eye openness, and movement
- A **standalone HTML report** readable by non-technical staff (no internet needed)
- An **integration JSON** ready for direct import into Sentio Mind's `person_database`

### Energy Score Formula
```
energy_score = brightness × 0.35 + eye_openness × 0.30 + movement × 0.35
```

| Component | Method |
|-----------|--------|
| Brightness | Grayscale mean pixel value scaled 0–100 |
| Eye Openness | MediaPipe Face Mesh landmark ratio scaled 0–100 |
| Movement | Dense optical flow (Farneback) magnitude scaled 0–100 |

---

## Demo

```
Step 1 — loading known faces ...
  Loaded: AARAV (1 encoding(s))
  Loaded: Anamika (1 encoding(s))
  ...
  10 persons loaded

Step 2 — extracting keyframes ...
  20 frames extracted

Step 3 & 4 — detecting + scoring faces ...
  Total face detections across all frames: 263

Step 5 — aggregating per-person ...
Step 6 — writing report.html ...
Step 7 — writing integration_output.json ...

=======================================================
  Finished in 110s
  Persons found: 41
    AARAV          energy  57.4  (moderate)
    Anamika        energy  57.3  (moderate)
    Harshita       energy  57.2  (moderate)
    ...
=======================================================
```

---

## Project Structure

```
sentio-identity/
│
├── solution.py                  ← Main script (run this)
├── report.html                  ← Generated HTML report (offline)
├── integration_output.json      ← Generated Sentio Mind JSON
├── demo.mp4                     ← Screen recording of working demo
├── README.md                    ← This file
│
├── known_faces/                 ← Reference photos (you provide)
│   ├── AARAV.jpg
│   ├── Anamika.jpg
│   ├── Harshita.jpg
│   └── ...                      (filename = student name, no spaces issues)
│
└── video_sample_1.mov           ← Input CCTV video (you provide)
```

> ⚠️ `known_faces/` photos must be **clear, front-facing, well-lit** for best matching accuracy.

---

## Setup on macOS

### Prerequisites
- macOS 11.0+ (works on both Intel and Apple Silicon M1/M2/M3)
- Homebrew installed → if not: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

---

### Step 1 — Install system dependencies
```bash
brew install cmake
brew install python@3.11
```

---

### Step 2 — Clone the repository
```bash
git clone https://github.com/Sentiodirector/sentio-poc-identity-energy.git
cd sentio-poc-identity-energy
```

---

### Step 3 — Create virtual environment with Python 3.11
```bash
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
```

> ⚠️ **Important**: Use Python 3.11 specifically. Python 3.12+ breaks several dependencies.

---

### Step 4 — Install Python dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install in this exact order
pip install cmake
pip install dlib --no-cache-dir
pip install face_recognition opencv-python mediapipe numpy==1.26.4 Pillow==10.3.0
```

> 💡 If `dlib` fails, run: `pip uninstall cmake -y` then retry `pip install dlib --no-cache-dir`  
> This removes the conflicting pip cmake so brew's cmake takes over.

---

### Step 5 — Verify installation
```bash
python -c "
import face_recognition, cv2, mediapipe, numpy
from PIL import Image
print('✅ All libraries loaded successfully!')
print('   OpenCV:', cv2.__version__)
print('   MediaPipe:', mediapipe.__version__)
print('   NumPy:', numpy.__version__)
"
```

---

## Setup on Windows

### Prerequisites
- Windows 10/11
- Python 3.11 from [python.org](https://www.python.org/downloads/release/python-3119/) — **check "Add to PATH"** during install
- Visual Studio Build Tools (needed for dlib)

---

### Step 1 — Install Visual Studio Build Tools
Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/  
During install, select **"Desktop development with C++"**

---

### Step 2 — Install CMake
Download from: https://cmake.org/download/  
During install, select **"Add CMake to the system PATH for all users"**

Verify:
```cmd
cmake --version
```

---

### Step 3 — Clone the repository
```cmd
git clone https://github.com/Sentiodirector/sentio-poc-identity-energy.git
cd sentio-poc-identity-energy
```

---

### Step 4 — Create virtual environment
```cmd
python -m venv venv
venv\Scripts\activate
```

---

### Step 5 — Install Python dependencies
```cmd
pip install --upgrade pip
pip install cmake
pip install dlib --no-cache-dir
pip install face_recognition opencv-python mediapipe numpy==1.26.4 Pillow==10.3.0
```

> 💡 If dlib still fails on Windows, download a pre-built wheel:  
> https://github.com/z-mahmud22/Dlib_Windows_Python3.x  
> Then: `pip install dlib-19.24.6-cp311-cp311-win_amd64.whl`

---

### Step 6 — Verify installation
```cmd
python -c "import face_recognition, cv2, mediapipe, numpy; print('All good!')"
```

---

## Running the Solution

### 1. Prepare your files
```
sentio-identity/
├── solution.py
├── video_sample_1.mov     ← place your video here
└── known_faces/
    ├── StudentName1.jpg   ← one photo per student
    ├── StudentName2.jpg   ← filename = their exact name
    └── ...
```

### 2. (Optional) Edit config
Open `solution.py` and adjust lines 20–24:
```python
SCHOOL_NAME     = "Your School Name"   # shown in report header
MATCH_THRESHOLD = 0.6                  # 0.6 = good for CCTV; lower = stricter
MAX_KEYFRAMES   = 20                   # increase for longer videos
```

### 3. Run
```bash
# macOS/Linux
python solution.py

# Windows
python solution.py
```

### 4. View outputs
```bash
# macOS
open report.html

# Windows
start report.html
```

---

## Understanding the Output

### report.html
A fully offline single-page report showing:
- Profile photo (best frame extracted from video)
- Name + matched/unknown badge
- Energy score (0–100) with color-coded verdict
- Breakdown bars for brightness, eye openness, movement

**Verdict scale:**
| Score | Verdict |
|-------|---------|
| ≥ 75 | 🟢 High |
| 50–74 | 🟡 Moderate |
| < 50 | 🔴 Low |

---

### integration_output.json
Follows the exact Sentio Mind `person_database` schema:
```json
{
  "source": "p1_identity_energy",
  "school": "Demo School",
  "date": "2025-03-17",
  "video_file": "video_sample_1.mov",
  "total_persons_matched": 10,
  "total_persons_unknown": 31,
  "processing_time_sec": 110.14,
  "persons": [
    {
      "person_id": "DEMOSH_P0001",
      "name": "AARAV",
      "matched": true,
      "match_confidence": 0.82,
      "profile_image_b64": "...",
      "frames_detected": 8,
      "energy_score": 57.4,
      "energy_breakdown": {
        "face_brightness": 61.2,
        "eye_openness": 50.0,
        "movement_activity": 60.1
      },
      "verdict": "moderate",
      "first_seen_frame": 42,
      "last_seen_frame": 310
    }
  ]
}
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KNOWN_FACES_DIR` | `known_faces/` | Folder with reference photos |
| `VIDEO_PATH` | `video_sample_1.mov` | Input video file |
| `SCHOOL_NAME` | `Demo School` | Shown in report and JSON |
| `MATCH_THRESHOLD` | `0.6` | Face match sensitivity (0.4–0.7 recommended) |
| `MAX_KEYFRAMES` | `20` | Frames sampled from video (increase for longer videos) |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No faces loaded` | Ensure photos in `known_faces/` are clear, front-facing, well-lit JPEGs |
| `dlib install fails on Mac` | Run `pip uninstall cmake -y` then `pip install dlib --no-cache-dir` |
| `dlib install fails on Windows` | Use pre-built wheel from z-mahmud22/Dlib_Windows_Python3.x |
| `Video file not found` | Make sure `video_sample_1.mov` is in the same folder as `solution.py` |
| Too many UNKNOWNs | Increase `MATCH_THRESHOLD` to `0.65`, ensure reference photos are high quality |
| Very slow processing | Normal — 20 frames takes ~2 min on CPU. Reduce `MAX_KEYFRAMES` to speed up |
| `mediapipe version error` | Run `pip install mediapipe` without version pin |
| `ModuleNotFoundError` | Make sure venv is activated: `source venv/bin/activate` (Mac) or `venv\Scripts\activate` (Windows) |
| Python 3.12/3.13 errors | Recreate venv with Python 3.11 specifically |

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `face_recognition` | 1.3.0 | Face encoding + identity matching |
| `opencv-python` | 4.11+ | Video processing, CLAHE, optical flow |
| `mediapipe` | 0.10.30+ | Face Mesh for eye openness detection |
| `numpy` | 1.26.4 | Numerical operations |
| `Pillow` | 10.3.0 | Image handling |
| `dlib` | 19.24+ | Underlying face detection (used by face_recognition) |

---

## Deliverables

| # | File | Description |
|---|------|-------------|
| 1 | `solution.py` | Complete working Python script |
| 2 | `report.html` | Self-contained offline HTML report |
| 3 | `integration_output.json` | Sentio Mind-compatible person database |
| 4 | `demo.mp4` | Screen recording of full working demo (<2 min) |

---

## License

Built for IIT Mandi × Sentio Mind Assignment · 2025
