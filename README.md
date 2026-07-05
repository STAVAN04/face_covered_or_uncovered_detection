# Face Covered vs Uncovered Detection (YOLO)

![Python](https://img.shields.io/badge/Python-3.10-blue) ![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-green) ![License](https://img.shields.io/badge/License-MIT-yellow) ![Status](https://img.shields.io/badge/Status-Active-success)

A real-time YOLO-based computer vision system that classifies whether a person's face is **visible** or **obstructed** — by a mask, sunglasses, cloth, hand, hood, or any other covering. Built for security and access-control use cases: ATMs, restricted entry points, and CCTV-monitored areas where a clear face is required.

Works on webcam feed, video files, and CCTV-style footage. Runs in real time.

## Problem

In security-sensitive environments, a person's face needs to remain visible to the camera. This model classifies each detected face as:

- **uncovered** — face is clearly visible
- **covered** — face is fully or partially obstructed (any object blocking key facial features counts)

If the full face isn't visible, it's treated as covered — this is a deliberately strict definition, appropriate for a security context.

## Dataset

- 3,000+ manually annotated images, YOLO format
- Collected across varied real-world conditions: rain, daylight, indoor/outdoor, CCTV angles, kiosks, shops, low-light frames
- Three classes: `covered`, `uncovered`, `background` (people/faces not relevant to classification)
- Deliberately includes hard cases: sunglasses, cloth masks, a phone held up to the face, hoodies, caps

## Training

| | |
|---|---|
| Model | YOLO |
| Epochs | 100 |
| Image size | 640×640 |
| Batch size | 16 |
| Optimizer | Adam |
| Training notebook | `train_model.ipynb` |
| Weights | `best.pt` |

## Results

| Metric | Value |
|---|---|
| mAP@50 | ~0.67 |
| mAP@50-95 | ~0.35 |
| Peak F1 | 0.67 at confidence 0.238 |

mAP@50 is solid for a three-class problem on real-world CCTV-style footage with heavy variation in lighting and angle. mAP@50-95 is more modest, which is expected here — it's a stricter metric and this dataset includes intentionally hard cases (low light, partial occlusion, unusual objects). Precision and recall both move with the confidence threshold as usual; the F1-confidence curve below shows where they balance.

**Training/validation curves:**
![results](runs/train/results.png)

**F1-confidence curve:**
![F1 Curve](runs/train/F1_curve.png)

**Confusion matrix (counts):**
![Confusion Matrix](runs/train/confusion_matrix.png)

**Confusion matrix (normalized):**
![Confusion Matrix Normalized](runs/train/confusion_matrix_normalized.png)

Reading the normalized matrix: covered → predicted covered at ~65%, uncovered → predicted uncovered at ~71%, background classified correctly at ~75%. 

## Limitations

- mAP@50-95 suggests bounding box localization could be tighter, particularly in low-light frames
- Performance drops in low-light environments and with heavily occluded faces.
- Small or distant faces are occasionally missed.
- Extreme camera angles and motion blur can reduce classification accuracy.

## Run it

```bash
git clone https://github.com/STAVAN04/Face-Covered-vs-Uncovered-Detection.git

cd Face-Covered-vs-Uncovered-Detection

pip install -r requirements.txt
python predict.py
```

`predict.py` runs inference on `<webcam / a video file — specify which, and how to point it at a file if applicable>`.

### Dependencies

```
ultralytics: 8.2.50
opencv-python: 4.8.0.76
PyTorch: 2.3.0
numpy: 1.25.2
```

## Example output

<p align="center">
  <img src="runs/train/val_batch2_pred.jpg" width="45%" alt="Predicted Results">
  &nbsp;&nbsp;
  <img src="runs/train/val_batch2_labels.jpg" width="45%" alt="Ground Truth Labels">
</p>

Left: model predictions. Right: ground-truth labels, for comparison.

## Next Steps (In Progress)

This project is actively being improved. Planned work:

- [ ] Expand low-light and nighttime CCTV training data to close the mAP@50-95 gap
- [ ] Add hard-negative mining for objects frequently confused with face coverings
- [ ] Tighten bounding box localization (current mAP@50-95 suggests room to improve here)
- [ ] Package as a lightweight model for edge/embedded deployment (e.g. Raspberry Pi, Jetson Nano)
- [ ] Add a simple live dashboard/alert system for real-time monitoring use cases


---
Built by [Stavan](https://github.com/STAVAN04)
