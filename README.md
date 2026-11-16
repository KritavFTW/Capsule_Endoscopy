# Capsule Frame Review

A small web app that detects lesions in wireless capsule endoscopy frames using a fine-tuned YOLO11s model.
Upload a frame, adjust the confidence threshold, and see boxed findings ranked by confidence.

> **Research demo only. Not a medical device. Do not use it to diagnose or treat anyone.**

## Run it

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                   # open http://127.0.0.1:5000
```

Needs `capsule_best.pt` in the project folder. `model_info.json` (optional) sets the default threshold.

## Model

- Dataset: SEE-AI capsule endoscopy (`capsuleyolo/kyucapsule` on Kaggle), 18,481 images, 12 lesion classes
- Base model: COCO-pretrained YOLO11s, fine-tuned at 640 px with Ultralytics
- Split: blocks of 100 consecutive frames kept together, with byte-identical images in the same split, to limit near-duplicate leakage between train and test (a blocked, near-duplicate-controlled split, not patient-level)

| Split | mAP@0.5 | mAP@0.5:0.95 |
|-------|---------|--------------|
| Test  | TODO    | TODO         |

Rare classes (for example SMT and diverticulum) have only a handful of test boxes, so their per-class scores are very noisy.

## Limitations

- Frames with no labeled lesion are common, and the model can miss lesions or raise false alarms
- The split is not patient-level, so real-world performance on new patients may be lower
- Trained and evaluated on one dataset; not validated clinically

## Data and licence

The dataset is not included. Get it from Kaggle and check its licence before reusing it.