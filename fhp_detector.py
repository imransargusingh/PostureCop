"""
Forward Head Posture (FHP) detector using Ultralytics YOLO-Pose (side view, RIGHT profile).

Right profile assumption:
- Person is facing LEFT in the image
- "Forward head" = head landmark moves LEFT of shoulder -> ear_x < shoulder_x
- We define forward score as: fhp_px = -(ear_x - shoulder_x)  => positive = forward

Outputs:
- fhp_norm (recommended): forward offset normalized by torso length
- cva_deg (secondary): "CVA proxy" angle above horizontal (smaller = more forward)

Controls:
- Press 'b' to (re)capture baseline for a few seconds in neutral posture
- Press 'q' or ESC to quit
"""

import time
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "YOLO26x-pose.pt"   # change if you have a custom pose model
CAM_INDEX = 0                   # webcam index

CONF_THR = 0.4                  # minimum keypoint confidence to accept a point
EMA_ALPHA = 0.2                 # smoothing factor (0-1), higher = more responsive
PERSIST_FRAMES = 10             # consecutive frames above threshold before flagging

BASELINE_SECONDS = 3.0          # capture duration for baseline
DELTA_NORM = 0.05               # forward offset threshold above baseline (normalized)
DELTA_CVA_DEG = 5.0             # CVA drop below baseline threshold

# COCO-17 keypoint indices used by YOLO-Pose (Ultralytics)
NOSE = 0
L_EAR, R_EAR = 3, 4
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12

# ----------------------------
# Helpers
# ----------------------------
def pick_best_point(kxy, kconf, idx_a, idx_b, thr=CONF_THR):
    """Pick the higher-confidence point among idx_a and idx_b, if either is above thr."""
    a_ok = kconf[idx_a] >= thr
    b_ok = kconf[idx_b] >= thr
    if a_ok and b_ok:
        return kxy[idx_a] if kconf[idx_a] >= kconf[idx_b] else kxy[idx_b]
    if a_ok:
        return kxy[idx_a]
    if b_ok:
        return kxy[idx_b]
    return None

def angle_between(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return None
    cosv = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def fhp_scores_right_profile(shoulder, head, hip=None):
    """
    shoulder, head, hip are np arrays of shape (2,) in image coords.
    RIGHT profile => forward is left => decreasing x => dx negative when forward.
    Define:
      dx = head_x - shoulder_x
      fhp_px = -dx  (positive means forward head)
      fhp_norm = fhp_px / torso_len  (torso_len = ||shoulder-hip||), if hip available
      cva_deg = atan2(|dy|, |dx|) in degrees; smaller means more forward
    """
    neck = head - shoulder
    dx, dy = float(neck[0]), float(neck[1])
    fhp_px = -dx

    fhp_norm = None
    if hip is not None:
        torso_len = float(np.linalg.norm(shoulder - hip))
        if torso_len > 1e-6:
            fhp_norm = fhp_px / torso_len

    cva_deg = float(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-9)))
    return fhp_px, fhp_norm, cva_deg

def choose_largest_person(results):
    """Return index of largest bounding box (area) in results.boxes."""
    if results.boxes is None or len(results.boxes) == 0:
        return None
    xyxy = results.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return int(np.argmax(areas))

def get_keypoints_for_person(results, person_idx):
    """Return (kxy, kconf) for the chosen person."""
    k = results.keypoints
    kxy = k.xy[person_idx].cpu().numpy()      # (17,2)
    kconf = k.conf[person_idx].cpu().numpy()  # (17,)
    return kxy, kconf

def draw_point(img, p, label=None):
    if p is None:
        return
    x, y = int(p[0]), int(p[1])
    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    if label:
        cv2.putText(img, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# ----------------------------
# Main
# ----------------------------
def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    # EMA state
    ema_fhp_norm = None
    ema_cva = None

    # Baseline state
    baseline_fhp_norm = None
    baseline_cva = None
    baseline_status = "Press 'b' to capture baseline (neutral posture)."

    # Baseline capture buffers
    capturing = False
    capture_start = 0.0
    buf_fhp_norm = []
    buf_cva = []

    # Persistence counter
    above_counter = 0

    # For display smoothing without hip (fallback)
    ema_fhp_px = None

    print("Running FHP detector. Right profile assumed (facing LEFT).")
    print("Controls: 'b' baseline, 'q' or ESC quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Inference
        results = model(frame, verbose=False)[0]
        person_idx = choose_largest_person(results)

        info_lines = []
        detected = False

        shoulder = head = hip = None
        fhp_px = fhp_norm = cva_deg = None

        if person_idx is not None and results.keypoints is not None:
            kxy, kconf = get_keypoints_for_person(results, person_idx)

            # Pick best shoulder, ear, hip
            shoulder = pick_best_point(kxy, kconf, L_SHO, R_SHO)
            ear = pick_best_point(kxy, kconf, L_EAR, R_EAR)
            hip = pick_best_point(kxy, kconf, L_HIP, R_HIP)
            nose = kxy[NOSE] if kconf[NOSE] >= CONF_THR else None

            if shoulder is not None:
                head = ear if ear is not None else nose

            if shoulder is not None and head is not None:
                fhp_px, fhp_norm, cva_deg = fhp_scores_right_profile(shoulder, head, hip)

                # Smooth scores
                if ema_fhp_px is None:
                    ema_fhp_px = fhp_px
                else:
                    ema_fhp_px = (1 - EMA_ALPHA) * ema_fhp_px + EMA_ALPHA * fhp_px

                if fhp_norm is not None:
                    if ema_fhp_norm is None:
                        ema_fhp_norm = fhp_norm
                    else:
                        ema_fhp_norm = (1 - EMA_ALPHA) * ema_fhp_norm + EMA_ALPHA * fhp_norm

                if ema_cva is None:
                    ema_cva = cva_deg
                else:
                    ema_cva = (1 - EMA_ALPHA) * ema_cva + EMA_ALPHA * cva_deg

                # Baseline capture mode
                if capturing:
                    if fhp_norm is not None:
                        buf_fhp_norm.append(fhp_norm)
                    buf_cva.append(cva_deg)

                    elapsed = time.time() - capture_start
                    baseline_status = f"Capturing baseline... {elapsed:.1f}/{BASELINE_SECONDS:.1f}s"

                    if elapsed >= BASELINE_SECONDS and len(buf_cva) >= 10:
                        baseline_cva = float(np.median(buf_cva))
                        baseline_fhp_norm = float(np.median(buf_fhp_norm)) if len(buf_fhp_norm) >= 10 else None

                        capturing = False
                        buf_fhp_norm.clear()
                        buf_cva.clear()

                        baseline_status = (
                            f"Baseline set: CVA={baseline_cva:.1f}°, "
                            + (f"FHP_norm={baseline_fhp_norm:.3f}" if baseline_fhp_norm is not None else "FHP_norm=N/A (no hip)")
                        )

                # Detection logic (prefer normalized if available)
                if baseline_cva is not None:
                    # Condition A: normalized forward offset above baseline
                    cond_norm = False
                    if baseline_fhp_norm is not None and ema_fhp_norm is not None:
                        cond_norm = ema_fhp_norm > (baseline_fhp_norm + DELTA_NORM)

                    # Condition B: CVA drops below baseline
                    cond_cva = False
                    if ema_cva is not None:
                        cond_cva = ema_cva < (baseline_cva - DELTA_CVA_DEG)

                    # Combine: either is enough; you can change to (cond_norm and cond_cva)
                    if cond_norm or cond_cva:
                        above_counter += 1
                    else:
                        above_counter = 0

                    detected = above_counter >= PERSIST_FRAMES

        # ----------------------------
        # Draw overlay
        # ----------------------------
        if shoulder is not None and head is not None:
            p1 = tuple(shoulder.astype(int))
            p2 = tuple(head.astype(int))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)
            draw_point(frame, shoulder, "shoulder")
            draw_point(frame, head, "head")
            if hip is not None:
                draw_point(frame, hip, "hip")

        # Text block
        y = 30
        def put(line):
            nonlocal y
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            y += 28

        put("Forward Head Posture (Right profile: facing LEFT)")
        put(baseline_status)

        if fhp_px is None:
            put("No reliable keypoints yet...")
        else:
            put(f"FHP_px (smoothed): {ema_fhp_px:7.1f}  (positive = forward)")
            if ema_fhp_norm is not None:
                put(f"FHP_norm (smoothed): {ema_fhp_norm:6.3f}")
            else:
                put("FHP_norm: N/A (hip not detected)")

            if ema_cva is not None:
                put(f"CVA proxy (smoothed): {ema_cva:5.1f} deg (smaller = forward)")

        if detected:
            cv2.putText(frame, "FHP DETECTED", (20, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        cv2.imshow("FHP Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        if key == ord('b'):
            # start baseline capture
            capturing = True
            capture_start = time.time()
            buf_fhp_norm.clear()
            buf_cva.clear()
            above_counter = 0
            baseline_status = f"Capturing baseline... 0.0/{BASELINE_SECONDS:.1f}s"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
