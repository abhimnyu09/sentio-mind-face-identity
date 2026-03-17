"""
solution.py
Sentio Mind · Project 1 · Named Face Identity + Energy Report

Run: python solution.py
"""

import cv2
import json
import base64
import time
import numpy as np
from pathlib import Path
from datetime import date

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------
KNOWN_FACES_DIR  = Path("known_faces")
VIDEO_PATH       = Path("video_sample_1.mov")
REPORT_HTML_OUT  = Path("report.html")
INTEGRATION_OUT  = Path("integration_output.json")

SCHOOL_NAME      = "Demo School"   # change to your school name
MATCH_THRESHOLD  = 0.6            # 0.55 works well for CCTV quality; lower = stricter
MAX_KEYFRAMES    = 20              # how many frames to sample from the video


# ---------------------------------------------------------------------------
# STEP 1 — Load reference photos
# ---------------------------------------------------------------------------

def load_known_faces(folder: Path) -> dict:
    """
    Read every image from the folder.
    The filename without extension is the person's name.
    Encode each face with face_recognition.
    Return: { "Arjun Mehta": [128-d encoding, ...], ... }
    """
    import face_recognition

    known = {}
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if not folder.exists():
        print(f"  WARNING: folder '{folder}' does not exist.")
        return known

    for img_path in sorted(folder.iterdir()):
        if img_path.suffix.lower() not in supported_exts:
            continue

        name = img_path.stem  # filename without extension = person name
        try:
            image = face_recognition.load_image_file(str(img_path))
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                print(f"  WARNING: No face found in '{img_path.name}', skipping.")
                continue
            if name not in known:
                known[name] = []
            known[name].extend(encodings)
            print(f"  Loaded: {name} ({len(encodings)} encoding(s))")
        except Exception as e:
            print(f"  WARNING: Could not process '{img_path.name}': {e}")

    return known


# ---------------------------------------------------------------------------
# STEP 2 — Extract keyframes from video
# ---------------------------------------------------------------------------

def extract_keyframes(video_path: Path, max_frames: int) -> list:
    """
    Open the video, pull up to max_frames evenly spaced frames.
    Apply CLAHE on each frame to help with CCTV lighting.
    Return: [(frame_index, numpy_array), ...]
    """
    frames = []

    if not video_path.exists():
        print(f"  ERROR: Video file '{video_path}' not found.")
        return frames

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video '{video_path}'.")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("  WARNING: Could not determine total frame count, reading sequentially.")
        total_frames = max_frames * 10  # fallback estimate

    step = max(1, total_frames // max_frames)

    # CLAHE for contrast enhancement (CCTV low-light improvement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    sampled_indices = list(range(0, total_frames, step))[:max_frames]

    for idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Apply CLAHE channel-wise in LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        frames.append((idx, enhanced))

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# STEP 3 — Detect and match faces in one frame
# ---------------------------------------------------------------------------

_unknown_counter = {"count": 0}

def detect_and_match(frame: np.ndarray, known: dict, threshold: float) -> list:
    """
    Detect all faces in frame, compare each against known encodings.
    Returns list of dicts with name, matched, confidence, bbox, face_crop.
    """
    import face_recognition

    detections = []

    # Convert BGR (OpenCV) → RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use HOG model for speed; use "cnn" for better accuracy if GPU available
    # Upsample 2x to help detect small (20px) faces
    face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2, model="hog")
    face_encodings_list = face_recognition.face_encodings(rgb_frame, face_locations)

    known_names = list(known.keys())
    all_known_encodings = []
    name_map = []  # parallel list: which name does each encoding belong to
    for name, enc_list in known.items():
        for enc in enc_list:
            all_known_encodings.append(enc)
            name_map.append(name)

    for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings_list):
        x, y, w, h = left, top, right - left, bottom - top

        # Ensure bbox is within frame bounds
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        if all_known_encodings:
            # Compute distances to all known encodings
            distances = face_recognition.face_distance(all_known_encodings, face_enc)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            confidence = round(float(1.0 - best_dist), 4)

            if best_dist <= threshold:
                matched_name = name_map[best_idx]
                matched = True
            else:
                _unknown_counter["count"] += 1
                matched_name = f"UNKNOWN_{_unknown_counter['count']:03d}"
                matched = False
                confidence = 0.0
        else:
            _unknown_counter["count"] += 1
            matched_name = f"UNKNOWN_{_unknown_counter['count']:03d}"
            matched = False
            confidence = 0.0

        detections.append({
            "name":       matched_name,
            "matched":    matched,
            "confidence": confidence,
            "bbox":       (x, y, w, h),
            "face_crop":  face_crop,
        })

    return detections


# ---------------------------------------------------------------------------
# STEP 4 — Energy components
# ---------------------------------------------------------------------------

def compute_face_brightness(face_crop: np.ndarray) -> float:
    """
    Grayscale mean pixel value scaled to 0–100.
    """
    if face_crop is None or face_crop.size == 0:
        return 50.0
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    return round(mean_val / 2.55, 2)  # scale 0–255 → 0–100


def compute_eye_openness(face_crop: np.ndarray) -> float:
    """
    Average of (eye height / eye width) for left and right eye, scaled 0–100.
    Uses MediaPipe Face Mesh to find eye landmarks.
    Falls back to 50.0 if detection fails.
    """
    if face_crop is None or face_crop.size == 0:
        return 50.0

    try:
        import mediapipe as mp

        mp_face_mesh = mp.solutions.face_mesh

        # MediaPipe landmark indices for eye corners / top / bottom
        # Left eye: 33 (outer), 133 (inner), 160 (top), 144 (bottom)
        # Right eye: 362 (outer), 263 (inner), 387 (top), 373 (bottom)
        LEFT_EYE_H  = [160, 144]  # top, bottom
        LEFT_EYE_W  = [33, 133]   # outer, inner
        RIGHT_EYE_H = [387, 373]
        RIGHT_EYE_W = [362, 263]

        h, w = face_crop.shape[:2]
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.3
        ) as face_mesh:
            results = face_mesh.process(rgb_crop)

        if not results.multi_face_landmarks:
            return 50.0

        lm = results.multi_face_landmarks[0].landmark

        def eye_ratio(h_idx, w_idx):
            y_top = lm[h_idx[0]].y * h
            y_bot = lm[h_idx[1]].y * h
            x_out = lm[w_idx[0]].x * w
            x_in  = lm[w_idx[1]].x * w
            eye_h = abs(y_bot - y_top)
            eye_w = abs(x_in  - x_out)
            if eye_w < 1e-6:
                return 0.0
            return eye_h / eye_w

        left_ratio  = eye_ratio(LEFT_EYE_H,  LEFT_EYE_W)
        right_ratio = eye_ratio(RIGHT_EYE_H, RIGHT_EYE_W)
        avg_ratio   = (left_ratio + right_ratio) / 2.0

        # Typical open-eye ratio ≈ 0.25–0.35; scale to 0–100
        # Clamp: 0.0 (fully closed) → 0, 0.4+ → 100
        score = min(100.0, (avg_ratio / 0.4) * 100.0)
        return round(score, 2)

    except Exception:
        return 50.0


def compute_movement(prev_frame, curr_frame: np.ndarray, bbox: tuple) -> float:
    """
    Dense optical flow magnitude in the face bounding box, scaled 0–100.
    Returns 0.0 if prev_frame is None.
    """
    if prev_frame is None:
        return 0.0

    if prev_frame is None or curr_frame is None:
        return 0.0

    try:
        x, y, w, h = bbox
        fh, fw = curr_frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)

        prev_crop = prev_frame[y1:y2, x1:x2]
        curr_crop = curr_frame[y1:y2, x1:x2]

        if prev_crop.size == 0 or curr_crop.size == 0:
            return 0.0

        # Resize to consistent size for flow computation
        size = (64, 64)
        prev_gray = cv2.cvtColor(cv2.resize(prev_crop, size), cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(cv2.resize(curr_crop, size), cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(magnitude))

        # Scale: typical mean magnitude 0–5 px/frame → 0–100
        score = min(100.0, mean_mag * 20.0)
        return round(score, 2)

    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# STEP 5 — Aggregate across all frames into per-person summaries
# ---------------------------------------------------------------------------

def _sharpness(img: np.ndarray) -> float:
    """Laplacian variance as a sharpness metric."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def aggregate_persons(all_detections: list) -> list:
    """
    Group detections by name, average energy components,
    compute energy_score, pick sharpest face crop.
    Returns list matching integration_output.json 'persons' schema.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for d in all_detections:
        groups[d["name"]].append(d)

    persons = []
    school_prefix = SCHOOL_NAME.replace(" ", "")[:6].upper()

    for pid_num, (name, dets) in enumerate(groups.items(), start=1):
        person_id = f"{school_prefix}_P{pid_num:04d}"

        brightnesses  = [d["brightness"]   for d in dets]
        eye_opennesses = [d["eye_openness"] for d in dets]
        movements      = [d["movement"]     for d in dets]
        confidences    = [d["confidence"]   for d in dets]
        frame_indices  = [d["frame_idx"]    for d in dets]

        avg_brightness  = round(float(np.mean(brightnesses)),  2)
        avg_eye         = round(float(np.mean(eye_opennesses)), 2)
        avg_movement    = round(float(np.mean(movements)),      2)

        energy_score = round(
            avg_brightness * 0.35 + avg_eye * 0.30 + avg_movement * 0.35, 2
        )

        # Pick sharpest crop as profile image
        crops = [d["face_crop"] for d in dets if d["face_crop"] is not None and d["face_crop"].size > 0]
        if crops:
            best_crop = max(crops, key=_sharpness)
        else:
            best_crop = np.zeros((240, 240, 3), dtype=np.uint8)

        profile_b64 = encode_b64(best_crop)

        matched   = dets[0]["matched"]
        avg_conf  = round(float(np.mean(confidences)), 4) if matched else 0.0

        persons.append({
            "person_id":          person_id,
            "name":               name,
            "matched":            matched,
            "match_confidence":   avg_conf,
            "profile_image_b64":  profile_b64,
            "frames_detected":    len(dets),
            "energy_score":       energy_score,
            "energy_breakdown": {
                "face_brightness":    avg_brightness,
                "eye_openness":       avg_eye,
                "movement_activity":  avg_movement,
            },
            "verdict":            verdict(energy_score),
            "first_seen_frame":   int(min(frame_indices)),
            "last_seen_frame":    int(max(frame_indices)),
        })

    # Sort: matched first, then by energy_score descending
    persons.sort(key=lambda p: (not p["matched"], -p["energy_score"]))
    return persons


def encode_b64(img: np.ndarray, size=(240, 240)) -> str:
    """Resize to size, encode as JPEG, return base64 string."""
    resized = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode("utf-8")


def verdict(score: float) -> str:
    """Returns 'high', 'moderate', or 'low'."""
    return "high" if score >= 75 else "moderate" if score >= 50 else "low"


# ---------------------------------------------------------------------------
# STEP 6 — HTML report
# ---------------------------------------------------------------------------

def _verdict_color(v: str) -> str:
    return {"high": "#22c55e", "moderate": "#f59e0b", "low": "#ef4444"}.get(v, "#94a3b8")


def generate_report(persons: list, output_path: Path):
    """
    Write a self-contained, fully offline HTML report.
    Includes profile photo, name, energy bar, breakdown, verdict per person.
    """
    today = str(date.today())
    total_matched = sum(1 for p in persons if p.get("matched"))
    total_unknown = sum(1 for p in persons if not p.get("matched"))

    # ── Person cards ────────────────────────────────────────────────────────
    cards_html = ""
    for p in persons:
        v = p["verdict"]
        vcolor = _verdict_color(v)
        score = p["energy_score"]
        eb = p["energy_breakdown"]
        img_src = f"data:image/jpeg;base64,{p['profile_image_b64']}"
        matched_badge = (
            f'<span style="background:#dcfce7;color:#166534;padding:2px 10px;'
            f'border-radius:99px;font-size:12px;font-weight:600;">✓ Matched</span>'
            if p["matched"] else
            f'<span style="background:#fee2e2;color:#991b1b;padding:2px 10px;'
            f'border-radius:99px;font-size:12px;font-weight:600;">? Unknown</span>'
        )

        def mini_bar(label, val, color):
            return f"""
            <div style="margin-bottom:6px;">
              <div style="display:flex;justify-content:space-between;font-size:12px;color:#64748b;margin-bottom:2px;">
                <span>{label}</span><span>{val:.1f}</span>
              </div>
              <div style="background:#e2e8f0;border-radius:4px;height:6px;">
                <div style="width:{min(val,100):.1f}%;background:{color};height:6px;border-radius:4px;"></div>
              </div>
            </div>"""

        cards_html += f"""
        <div style="background:#fff;border-radius:16px;box-shadow:0 2px 12px rgba(0,0,0,0.08);
                    padding:24px;display:flex;gap:20px;align-items:flex-start;border:1px solid #f1f5f9;">
          <img src="{img_src}" alt="{p['name']}"
               style="width:80px;height:80px;border-radius:12px;object-fit:cover;
                      flex-shrink:0;border:2px solid #e2e8f0;" />
          <div style="flex:1;min-width:0;">
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
              <h3 style="margin:0;font-size:16px;font-weight:700;color:#1e293b;">{p['name']}</h3>
              {matched_badge}
            </div>
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
              <span style="font-size:28px;font-weight:800;color:{vcolor};">{score:.1f}</span>
              <div>
                <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">Energy Score</div>
                <div style="font-size:13px;font-weight:600;color:{vcolor};text-transform:capitalize;">{v}</div>
              </div>
              <div style="margin-left:auto;text-align:right;font-size:11px;color:#94a3b8;">
                <div>Frames: {p['frames_detected']}</div>
                <div>ID: {p['person_id']}</div>
              </div>
            </div>
            <div style="background:#f8fafc;border-radius:8px;padding:10px;">
              {mini_bar('Brightness',  eb['face_brightness'],   '#3b82f6')}
              {mini_bar('Eye Openness', eb['eye_openness'],     '#8b5cf6')}
              {mini_bar('Movement',    eb['movement_activity'], '#f59e0b')}
            </div>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Sentio Mind — Energy Report · {SCHOOL_NAME}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f8fafc;
    color: #1e293b;
    line-height: 1.5;
  }}
  .header {{
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    color: #fff;
    padding: 32px 40px;
  }}
  .header h1 {{ font-size: 24px; font-weight: 800; margin-bottom: 4px; }}
  .header p  {{ font-size: 14px; color: #94a3b8; }}
  .stats {{
    display: flex; gap: 16px; padding: 24px 40px;
    flex-wrap: wrap;
  }}
  .stat {{
    background: #fff; border-radius: 12px; padding: 16px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); flex: 1; min-width: 140px;
    border: 1px solid #f1f5f9;
  }}
  .stat .val {{ font-size: 28px; font-weight: 800; color: #1e293b; }}
  .stat .lbl {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 16px;
    padding: 0 40px 40px;
  }}
  .section-title {{
    padding: 0 40px 12px;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
  }}
  .footer {{
    text-align: center;
    padding: 20px;
    font-size: 12px;
    color: #94a3b8;
    border-top: 1px solid #e2e8f0;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>⚡ Sentio Mind · Energy Report</h1>
  <p>{SCHOOL_NAME} &nbsp;·&nbsp; Generated {today} &nbsp;·&nbsp; Video: {VIDEO_PATH.name}</p>
</div>

<div class="stats">
  <div class="stat">
    <div class="val">{len(persons)}</div>
    <div class="lbl">Total Persons</div>
  </div>
  <div class="stat">
    <div class="val" style="color:#22c55e;">{total_matched}</div>
    <div class="lbl">Identified</div>
  </div>
  <div class="stat">
    <div class="val" style="color:#f59e0b;">{total_unknown}</div>
    <div class="lbl">Unknown</div>
  </div>
  <div class="stat">
    <div class="val">{round(sum(p['energy_score'] for p in persons)/max(len(persons),1), 1)}</div>
    <div class="lbl">Avg Energy Score</div>
  </div>
</div>

<p class="section-title">Person Profiles</p>
<div class="grid">
{cards_html}
</div>

<div class="footer">
  Sentio Mind · Named Face Identity &amp; Energy Report · {today}
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  report.html written ({output_path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# STEP 7 — Integration JSON
# ---------------------------------------------------------------------------

def write_integration_json(persons: list, output_path: Path,
                            video_name: str, processing_time: float):
    """
    Write integration_output.json following identity_energy.json exactly.
    """
    output = {
        "source": "p1_identity_energy",
        "school": SCHOOL_NAME,
        "date": str(date.today()),
        "video_file": video_name,
        "total_persons_matched": sum(1 for p in persons if p.get("matched")),
        "total_persons_unknown": sum(1 for p in persons if not p.get("matched")),
        "processing_time_sec": processing_time,
        "persons": persons,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  integration_output.json written ({output_path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    print("Step 1 — loading known faces ...")
    known = load_known_faces(KNOWN_FACES_DIR)
    if not known:
        print("  ERROR: no faces loaded. Check known_faces/ has images named like 'Arjun Mehta.jpg'")
        raise SystemExit(1)
    print(f"  {len(known)} persons: {', '.join(list(known.keys())[:6])}")

    print("Step 2 — extracting keyframes ...")
    frames = extract_keyframes(VIDEO_PATH, MAX_KEYFRAMES)
    print(f"  {len(frames)} frames extracted")

    print("Step 3 & 4 — detecting + scoring faces ...")
    all_detections = []
    prev_frame = None
    for frame_idx, frame in frames:
        detections = detect_and_match(frame, known, MATCH_THRESHOLD)
        for d in detections:
            d["frame_idx"]    = frame_idx
            d["brightness"]   = compute_face_brightness(d["face_crop"])
            d["eye_openness"] = compute_eye_openness(d["face_crop"])
            d["movement"]     = compute_movement(prev_frame, frame, d["bbox"])
        all_detections.extend(detections)
        prev_frame = frame

    print(f"  Total face detections across all frames: {len(all_detections)}")

    print("Step 5 — aggregating per-person ...")
    persons = aggregate_persons(all_detections)

    t1 = round(time.time() - t0, 2)

    print("Step 6 — writing report.html ...")
    generate_report(persons, REPORT_HTML_OUT)

    print("Step 7 — writing integration_output.json ...")
    write_integration_json(persons, INTEGRATION_OUT, str(VIDEO_PATH), t1)

    print()
    print("=" * 55)
    print(f"  Finished in {t1}s")
    print(f"  Persons found: {len(persons)}")
    for p in persons:
        print(f"    {p['name']:30s}  energy {p['energy_score']:5.1f}  ({p['verdict']})")
    print(f"  report.html              -> {REPORT_HTML_OUT}")
    print(f"  integration_output.json  -> {INTEGRATION_OUT}")
    print("=" * 55)
