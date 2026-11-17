# CapsuleCopilot: AI-Agent-Assisted Gastrointestinal Abnormality Detection in Wireless Capsule Endoscopy

CapsuleCopilot is an AI-assisted research framework for detecting gastrointestinal (GI) abnormalities in Wireless Capsule Endoscopy (WCE) images. It combines YOLO-based object detection, dataset auditing, data-leakage analysis, robustness evaluation, and rule-based AI agents to support a more reliable and reproducible medical-image analysis workflow.

The project includes an interactive **Flask-based web dashboard** for reviewing individual capsule-endoscopy frames. Users can upload an image, adjust the detection confidence threshold, and view detected abnormalities with bounding boxes and confidence scores.

> **Disclaimer:** This project is a research prototype, not a certified medical device. It is not intended to diagnose, treat, or replace the judgement of qualified medical professionals.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Models and Methodology](#models-and-methodology)
- [Dataset](#dataset)
- [Experimental Results](#experimental-results)
- [Flask Dashboard](#flask-dashboard)
- [AI Agent Workflow](#ai-agent-workflow)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Project Overview

Wireless Capsule Endoscopy allows doctors to examine the gastrointestinal tract using a small camera-equipped capsule. However, a single examination can generate thousands of images, making manual review time-consuming.

CapsuleCopilot aims to assist this process through deep-learning-based lesion detection and automated dataset analysis.

The framework focuses on:

- Detecting gastrointestinal abnormalities in capsule-endoscopy images.
- Reproducing and evaluating the RFC-YOLO architecture.
- Identifying duplicate images and annotation conflicts.
- Detecting possible data leakage between training and testing datasets.
- Comparing model performance under different image conditions.
- Using AI agents to audit experimental results and generate reports.
- Providing an interactive Flask dashboard for frame-level review.

## Key Features

### 1. Gastrointestinal Abnormality Detection
- YOLOv11n baseline detection model.
- COCO-pretrained YOLOv11n model.
- RFC-YOLO reproduction with additional attention mechanisms.
- Bounding-box localization and confidence-based detection.

### 2. Dataset Auditing
- Image and annotation validation.
- Exact duplicate detection using SHA-256 hashing.
- Near-duplicate detection using image embeddings.
- Label-conflict identification.
- Annotation sanitization.
- Dataset split integrity verification.

### 3. Data Leakage Analysis
- Random train-validation-test split.
- Near-duplicate-controlled blocked split.
- Deduplicated blocked split.
- Similarity-based leakage measurement.

### 4. Model Evaluation
- mAP@0.5.
- mAP@0.5:0.95.
- Precision, Recall, and F1-score.
- Per-class Average Precision.
- Confusion matrices.
- Paired bootstrap statistical analysis.

### 5. Robustness Analysis
Models are evaluated under different image conditions:

- Gaussian noise.
- Gaussian blur.
- Brightness changes.
- Contrast changes.
- JPEG compression.

### 6. AI-Agent-Based Auditing
Multiple rule-based agents coordinate dataset validation, leakage analysis, evaluation, robustness testing, and report generation.

### 7. Interactive Flask Dashboard
- Upload a single capsule-endoscopy frame.
- Adjust the confidence threshold.
- Detect and localize abnormalities.
- Display bounding boxes and confidence-ranked predictions.
- Review model output through a web interface.

---

## System Architecture

The project follows this general workflow:

```text
             WCE Image Dataset
                     |
                     v
          Dataset Validation
                     |
                     v
          Duplicate and Label Audit
                     |
                     v
          Leakage-Controlled Splitting
                     |
                     v
       +-------------+-------------+
       |             |             |
       v             v             v
    YOLOv11n    COCO YOLOv11n    RFC-YOLO
       |             |             |
       +-------------+-------------+
                     |
                     v
              Model Evaluation
                     |
                     v
             Robustness Testing
                     |
                     v
             AI Agent Auditing
                     |
                     v
          Reports and HTML Dashboard
                     |
                     v
             Flask Web Interface
                     |
                     v
          Capsule Frame Review Demo
```

---

## Models and Methodology

### RFC-YOLO Implementation

The project reproduces the major architectural components of RFC-YOLO using a YOLOv11n-based model.

| Component | Purpose |
|---|---|
| RFAConv | Receptive-field-aware feature extraction |
| C3k2-CBAM | Channel and spatial attention |
| Focal-MPDIoU | Improved bounding-box regression loss |

**RFAConv:** Replaces selected stride-2 convolution layers in the backbone to enhance receptive-field feature extraction.

**C3k2-CBAM:** Combines the C3k2 module with Convolutional Block Attention Module (CBAM) in the neck.

**Focal-MPDIoU:** Applies a focal weighting strategy to the MPDIoU-based bounding-box regression objective.

*Note: This is an implementation/reproduction based on the paper's described components. Exact architectural parity with the original authors' implementation has not been established.*

### Model Comparison

The following models are evaluated:

1. YOLOv11n trained from scratch.
2. COCO-pretrained YOLOv11n.
3. RFC-YOLO reproduction.

---

## Dataset

The project uses the **SEE-AI Wireless Capsule Endoscopy dataset**.

| Dataset information | Value |
|---|---:|
| Total images | 18,481 |
| Bounding boxes | 23,033 |
| Background-only images | 6,161 |
| Lesion categories | 12 |

The 12 categories include:

- Angiodysplasia
- Erosion
- Stenosis
- Lymphangiectasis
- Lymph follicle
- Submucosal tumor
- Polyp-like lesions
- Bleeding
- Diverticula
- Redness
- Foreign body
- Venous structures

---

## Experimental Results

The following results are taken from the notebook's recorded test-set evaluation. The models were evaluated on their corresponding leakage-controlled test splits.

### Model Performance

| Model | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| YOLOv11n Baseline | 53.11% | 31.20% |
| COCO-pretrained YOLOv11n | 61.14% | 37.03% |
| RFC-YOLO Reproduction | 52.20% | 29.88% |

**Observation:** In this recorded experiment, COCO-pretrained YOLOv11n achieved the highest test performance. RFC-YOLO did not outperform the baseline at mAP@0.5 in this single-seed reproduction.

### Dataset Leakage Audit

| Split | Test near-duplicate leakage (cosine similarity ≥ 0.95) |
|---|---:|
| Random split | 18.9% |
| Blocked split | 0.0% |
| Deduplicated blocked split | 0.0% |

The audit identified 121 exact-duplicate pairs, of which 107 contained annotation conflicts.

### Robustness Evaluation

Mean mAP@0.5 retention across the evaluated image corruptions:

| Model | Mean retention |
|---|---:|
| YOLOv11n Baseline | 94.94% |
| COCO-pretrained YOLOv11n | 95.15% |
| RFC-YOLO | 93.11% |

These values represent average performance retention relative to each model's clean-image result.

### Important Experimental Note

The experiment uses one training seed per model. Bootstrap confidence intervals measure test-set sampling variability, not variation across independent training runs. The split is near-duplicate-controlled, not patient-level. Therefore, the results should be interpreted as research findings rather than clinical performance claims.

---

## Flask Dashboard

CapsuleCopilot includes an interactive web interface for reviewing individual wireless capsule-endoscopy frames.

### Capsule Frame Review

The interface allows users to upload an image and run the trained detection model.

**Main interface features:**
- Single-image upload.
- Confidence threshold adjustment.
- Lesion detection.
- Bounding-box visualization.
- Confidence-ranked detection results.

### Dashboard Screenshot

Add the uploaded screenshot to your repository at:

`assets/capsule-frame-review.png`

Then display it here:

![CapsuleCopilot Frame Review Dashboard](assets/capsule-frame-review.png)

The notebook also generates a separate self-contained research dashboard at `reports/dashboard.html`, containing dataset statistics, duplicate analysis, leakage summaries, model comparisons, robustness results, and claim-verification information.

---

## AI Agent Workflow

CapsuleCopilot uses a deterministic, rule-based multi-agent system.

| Agent | Responsibility |
|---|---|
| PlannerAgent | Plans and tracks audit tasks |
| PaperClaimExtractor | Extracts claims from the research paper |
| DataAuditor | Checks dataset quality, duplicates, and label conflicts |
| SplitAuditor | Evaluates data leakage |
| EvaluationAuditor | Analyzes model performance |
| RobustnessAuditor | Evaluates performance under image corruptions |
| VerifierAgent | Verifies claims against saved evidence |
| ReportGenerator | Generates final reports |

The agents exchange information through saved experiment artifacts and JSON results. An LLM is not required for the core rule-based audit workflow.

---



---

## Installation and Setup

### Requirements

- Python 3.10 or compatible environment.
- PyTorch.
- Ultralytics.
- OpenCV.
- NumPy and Pandas.
- Scikit-learn.
- Matplotlib.
- Flask for the web dashboard.
- CUDA-enabled GPU recommended for model training.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/CapsuleCopilot.git
cd CapsuleCopilot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download or obtain access to the SEE-AI WCE dataset and configure its path in the notebook.

The original notebook is designed for Kaggle. Attach the dataset, enable GPU acceleration, and execute the notebook cells in sequence.

---

## How to Run

### Research Pipeline

1. Open `rfc_yolo_final.ipynb` in Kaggle.
2. Attach the SEE-AI capsule-endoscopy dataset.
3. Enable GPU acceleration.
4. Execute the notebook cells in order.
5. Review dataset audit and split-integrity reports.
6. Train the YOLOv11n, COCO-pretrained, and RFC-YOLO models.
7. Run test evaluation and robustness analysis.
8. Generate the final audit report and dashboard.

### Flask Dashboard

From the Flask application directory, run:

```bash
python app.py
```

Open the local address displayed by Flask in your browser, commonly:

```text
http://127.0.0.1:5000/
```

Upload a capsule-endoscopy frame, set the confidence threshold, and select **Find lesions** to view the detection output.

*The Flask command assumes your application entry point is named `app.py` and configured to start the server.*

---

## Limitations

- The project is an experimental research prototype.
- Model evaluation uses a single training seed.
- The dataset split is not patient-level.
- Reproduced RFC-YOLO performance differs from the paper's reported results.
- The dataset may not represent the full diversity of real-world clinical examinations.
- Frame-level predictions are not equivalent to a medical diagnosis.
- Further validation on independent clinical datasets is required.

---

## Future Improvements

- Evaluate models across multiple random seeds.
- Introduce patient-level dataset splitting where patient identifiers are available.
- Improve detection accuracy for underrepresented lesion classes.
- Optimize trained models for faster inference.
- Extend the Flask dashboard with experiment-history and comparison views.
- Integrate explainability techniques such as Grad-CAM.
- Validate the framework on additional WCE datasets.
- Conduct evaluation with clinical experts.

---

## Technologies Used

**Programming:** Python

**Deep Learning:** PyTorch, Ultralytics YOLO

**Computer Vision:** OpenCV, PIL

**Data Analysis:** NumPy, Pandas, Scikit-learn, Matplotlib

**Web Application:** Flask, HTML, CSS

**Research and Experimentation:** Kaggle, Jupyter Notebook, CUDA

---

## Authors

**CapsuleCopilot — AI-Agent-Assisted Gastrointestinal Abnormality Detection in Wireless Capsule Endoscopy**

Developed as an academic research project.

---

## Acknowledgements

- The authors of the RFC-YOLO research paper.
- The SEE-AI dataset contributors.
- The open-source developers of PyTorch, Ultralytics, and other libraries used in this project.

---

**Disclaimer:** CapsuleCopilot is intended solely for academic and research purposes. It must not be used for medical diagnosis or treatment decisions.