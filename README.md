# DeepPlastic - Marine Plastic Detection using Deep Learning

> **Real-time detection of epipelagic plastic debris** using YOLOv4 and YOLOv5 object detection models.

![Inference Demo](https://media.giphy.com/media/KCglrSW1FKhkNS6B5G/giphy.gif)

---

## Overview

This project implements an automated plastic detection system for marine environments using state-of-the-art YOLO (You Only Look Once) object detection. The system can identify floating plastic debris in real-time from underwater and surface-level imagery.

**Paper:** [DeepPlastic: A Novel Approach to Detecting Epipelagic Bound Plastic Using Deep Visual Models](https://arxiv.org/pdf/2105.01882.pdf)

**Video Demo:** [YouTube - DeepPlastic Results](https://youtu.be/8zBdFxaK4Os)

---

## Dataset - DeepTrash

| Property | Details |
|---|---|
| **Total Images** | 3,209 |
| **Training Set** | 1,900 images (60%) |
| **Validation Set** | 637 images (20%) |
| **Test Set** | 637 images (20%) |
| **Classes** | 1 (`trash_plastic`) |
| **Image Size** | 416 x 416 pixels |
| **Format** | YOLO (bounding box annotations) |
| **Source** | [Roboflow](https://roboflow.com/) export |

### Data Sources
- **Field images** from Lake Tahoe, San Francisco Bay, and Bodega Bay (California)
- **Internet images** (<20%) scraped from Google Images
- **Deep sea images** from [JAMSTEC JEDI dataset](http://www.godac.jamstec.go.jp/)

---

## Models

### YOLOv5 (Recommended)
- **Architecture:** Custom YOLOv5s (small) - optimized for speed and accuracy
- **Framework:** [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) (PyTorch)
- **Training:** 500 epochs, batch size 16, image size 416x416
- **Weights:** [Download best.pt](https://drive.google.com/file/d/14mBOhtLrE2d3hudqjwBZmawKAvTF4zxS/view?usp=sharing)
- **Notebook:** [`DeepPlastic-master/Github_Yolov5.ipynb`](DeepPlastic-master/Github_Yolov5.ipynb)

### YOLOv4-Tiny
- **Architecture:** Custom YOLOv4-Tiny detector - lightweight for edge deployment
- **Framework:** [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) (C/CUDA)
- **Training:** 10,000 max batches, image size 416x416
- **Weights:** [Download best.weights](https://drive.google.com/file/d/1YOTtZ2cHbqgxHukzLp01OVsUoa2CwwXs/view?usp=sharing)
- **Notebook:** [`DeepPlastic-master/Github_Yolov4.ipynb`](DeepPlastic-master/Github_Yolov4.ipynb)

---

## Project Structure

```
plastic/
├── README.md                  # This file
├── data.yaml                  # YOLOv5 dataset configuration
│
├── DeepPlastic-master/        # Source code & notebooks
│   ├── Github_Yolov5.ipynb    # YOLOv5 training notebook (Google Colab)
│   ├── Github_Yolov4.ipynb    # YOLOv4 training notebook (Google Colab)
│   ├── README.md              # Original project README
│   └── LICENSE                # MIT License
│
├── yolo5/                     # YOLOv5 PyTorch format dataset
│   ├── data.yaml              # Dataset config for yolo5 subfolder
│   ├── train/
│   │   ├── images/            # Training images (.jpg)
│   │   └── labels/            # Training labels (.txt, YOLO format)
│   ├── valid/
│   │   ├── images/            # Validation images
│   │   └── labels/            # Validation labels
│   └── test/
│       ├── images/            # Test images
│       └── labels/            # Test labels
│
├── yolo4/                     # YOLOv4 Darknet format dataset
│   ├── train/                 # Flat structure: .jpg + .txt side-by-side
│   ├── valid/
│   └── test/
│
├── train/                     # Additional flat-format dataset copy
├── valid/
└── test/
```

---

## Quick Start

### Option 1: Google Colab (Recommended)
1. Open the desired notebook in Google Colab:
   - [YOLOv5 Notebook](DeepPlastic-master/Github_Yolov5.ipynb) 
   - [YOLOv4 Notebook](DeepPlastic-master/Github_Yolov4.ipynb)
2. Go to `Runtime → Change Runtime Type → GPU`
3. Follow the step-by-step cells in the notebook

### Option 2: Local Setup (YOLOv5)
```bash
# Clone YOLOv5
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

# Train
python train.py --img 416 --batch 16 --epochs 500 \
    --data ../data.yaml \
    --cfg models/yolov5s.yaml \
    --weights yolov5s.pt \
    --name deepplastic

# Inference
python detect.py --weights runs/train/deepplastic/weights/best.pt \
    --img 416 --source ../yolo5/test/images/ --conf-thres 0.4
```

---

## Annotation Format

Labels use standard YOLO format (one `.txt` file per image):
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized to [0, 1]. Single class: `0` = `trash_plastic`.

---

## Results

The model achieves high precision in detecting floating plastic debris across diverse marine environments.

![Results](https://github.com/gautamtata/DeepPlastic/blob/master/results.png)

---

## Citation

```bibtex
@misc{tata2021deepplastic,
    title={DeepPlastic: A Novel Approach to Detecting Epipelagic Bound Plastic Using Deep Visual Models}, 
    author={Gautam Tata and Sarah-Jeanne Royer and Olivier Poirion and Jay Lowe},
    year={2021},
    eprint={2105.01882},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](DeepPlastic-master/LICENSE) for details.

## Resources

- **Dataset Download:** [Google Drive](https://drive.google.com/drive/folders/1fsS_u2QpbRGynYkP6-D6cfvq8r0hpjXI?usp=sharing)
- **arXiv Paper:** [2105.01882](https://arxiv.org/abs/2105.01882)
- **Original Repository:** [gautamtata/DeepPlastic](https://github.com/gautamtata/DeepPlastic)
