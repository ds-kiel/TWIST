# TWIST: Trains under Weather, Illumination, and Seasonal Time

## 📌 Overview

**TWIST** is a real-world dataset for train detection designed to improve robustness of vision-based railway monitoring systems under diverse environmental conditions.
Vision-based railway monitoring systems often fail in real-world deployments due to limited training data diversity. TWIST addresses this gap by providing a dataset collected across **multiple seasons**, capturing:

- 🌧️ Rain  
- ❄️ Snow  
- 🌫️ Fog  
- 🌙 Low-light & night  
- ☀️ Glare & daylight  
- 🚄 Motion blur & varying train speeds  

The dataset is specifically designed for **robust train detection in real operational environments**.

📥 [*Get the TWIST Dataset here*](https://zenodo.org/records/19472084?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImU3ZGEwMmZkLTZlNGUtNGI3Mi1iYzJlLWY2YjNiYzc2ODJjNyIsImRhdGEiOnt9LCJyYW5kb20iOiIxNjg5MjUyN2NjMGRlYzNjZDYzZjhlYWVkNGU5MzljMSJ9.NKBTdc8WEAofOfK601R6i70IzxCIpGds3q4uKSCWmqUMY32ZjE9bYkayyqDimaq4dPy8C1GsGCh1QLSkiRdziw)

---

## 📊 Dataset Statistics

- **Total Images:** ~38,000  
- **Resolution:** 640 × 480  
- **Binary Labeled Images:** ~10,000  
- **Detailed Annotations:** ~1,493 images  

### Label Types

1. **Binary Detection**
   - Train / No Train  

2. **Multi-class (subset)**
   - Locomotive  
   - Wagon  
   - Freight Car  
   - High-Speed Train  

---

## 🌍 Key Features

- ✅ Real-world railway environment  
- ✅ Diverse weather conditions  
- ✅ Day & night coverage  
- ✅ Motion blur and distance variation  
- ✅ Fixed-camera realistic deployment setup  
- ✅ YOLO-compatible annotations  

---

## 🏷️ Annotation Format

Annotations follow the **YOLO format**:

```
<class_id> <x_center> <y_center> <width> <height>
```

- All values are normalized between 0 and 1  
- Compatible with YOLOv5, YOLOv8, and other frameworks  

---

## 🚀 Benchmark Results

We benchmark TWIST using **YOLOv8s**:

| Setup | Training Data        | binary classification | mAP (%) | AP (Train) (%) |
|-------|---------------------|--------|--------|----------------|
| A     | COCO                |❌| 60.4   | 75.9           |
| B     | COCO + TWIST        |❌| 60.8   | 89.1           |
| C     | COCO → TWIST (test) |✔️| 81.1   | 81.1           |
| D     | COCO + TWIST        |✔️| 99.0   | 99.0           |
| E     | TWIST only          |✔️| 99.0   | 99.0           |

📈 **Key Insight:**  
Training with TWIST significantly improves detection performance, especially under challenging conditions.

---

## ⚙️ Training Example (YOLOv8)

```bash
pip install ultralytics

yolo detect train \
  data=twist.yaml \
  model=yolov8s.yaml \
  epochs=100 \
  imgsz=640
```

---

## 🧪 Data Split

Default split used in the paper:

- **70%** Training  
- **20%** Validation  
- **10%** Testing  

---

## 🖥️ Edge Deployment

Models trained on TWIST were successfully deployed on:

- NVIDIA Jetson Orin Nano  
- **Performance:** 25–30 FPS  

This demonstrates suitability for **real-time edge AI applications**.

---

## 🔍 Use Cases

- Railway safety monitoring  
- Worker alert systems  
- Autonomous inspection systems  
- Edge AI deployment research  
- Robust object detection benchmarking  

---

## ⚠️ Limitations

- Limited viewpoint diversity (fixed camera setup)  
- Fewer extreme weather edge cases  
- Slight imbalance in night vs. day data  
- Single-annotator labeling (possible minor bias)  

---

## 📜 Citation

If you use this dataset, please cite:

```bibtex
@article{twist2026,
  title={TWIST: Trains under Weather, Illumination, and Seasonal Time},
  author={Ali, Momin and Stenger, Andre and Arkenberg, Til and Harms, Laura and Landsiedel, Olaf},
  journal={},
  year={2026}
}
```

---

## 🤝 Acknowledgements

- Kiel University  
- ZÖLLNER Signal GmbH  
- Hamburg University of Technology  
- United Nations University (Hub on Engineering to Face Climate Change)  

---

## 📬 Contact

For questions or collaborations:

📧 momin.ali@cs.uni-kiel.de  

---

## ⭐ Contributing

Contributions are welcome! Please open an issue or submit a pull request.
