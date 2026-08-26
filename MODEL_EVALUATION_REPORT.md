# Raritone VTON Model Evaluation Report

## 1. Evaluation Objective

The purpose of this evaluation was to systematically test the current Raritone virtual try-on pipeline, identify failure cases, measure latency, evaluate output quality, and validate improvements made to the input-quality pipeline.

The evaluated pipeline is:

```text
Person Image + Garment Image
        ↓
Input Validation
        ↓
Person Preprocessing
        ↓
Pose Detection
        ↓
Person Segmentation
        ↓
Garment Preprocessing
        ↓
Quality Gate
        ↓
Pose-Aware Try-On Inference
        ↓
Output Validation
        ↓
Result
```

The current model version is:

```text
tryon-v1-baseline
```

This is a baseline pose-aware 2D compositing prototype and not a photorealistic generative VTON model.

---

# 2. Evaluation Dataset

## Source Images

The evaluation dataset was created using:

- 3 authorized person images
- 2 authorized garment images

Garment categories currently covered:

- T-shirt
- Shirt / Top

## Controlled Evaluation Conditions

Each person-garment pair was evaluated under four controlled image conditions:

1. Original
2. Low resolution
3. Dark lighting
4. Bright lighting

This produced:

```text
3 person images
×
2 garment images
×
4 input conditions
=
24 evaluation combinations
```

The evaluation metadata is stored in:

```text
evaluation/metadata.csv
```

Generated results are stored in:

```text
evaluation/generated/
```

---

# 3. Automated Evaluation Pipeline

The automated evaluation runner is:

```text
evaluation/run_evaluation.py
```

For each test case it performs:

1. Load person image
2. Load garment image
3. Apply controlled input condition
4. Run person preprocessing
5. Run garment preprocessing
6. Run quality gate
7. Extract pose and mask quality
8. Run try-on inference
9. Validate output
10. Save generated result
11. Record processing time
12. Record success/failure

The final results are saved in:

```text
evaluation/evaluation_results.csv
```

---

# 4. Automated Metrics

The evaluation pipeline records:

- Input resolution
- Output resolution
- Output availability
- Person processing time
- Garment processing time
- Inference time
- Total processing time
- Pose quality
- Person mask quality
- Garment quality
- Person-mask coverage
- Alignment indicator
- Success/failure
- Failure reason

Human visual-quality fields are also included for:

- Visual quality score
- Alignment score
- Garment preservation score
- Artifact score

These visual scores use a 1–5 rating scale and are not replaced by arbitrary automated realism metrics.

---

# 5. Baseline Evaluation Results

Before pipeline improvements:

```text
Tests run: 24
Successful: 16
Failed: 8
Failure rate: 33.33%

Average inference time: 1.2138 seconds
Average total processing time: 4.1152 seconds
```

All eight rejected cases were associated with:

```text
PARTIAL_BODY
```

The affected source image was:

```text
person2.jpg
```

The same input failed across both garment types and all four image variants.

---

# 6. Failure Analysis

Failure analysis was generated using:

```text
evaluation/analyze_failures.py
```

Output:

```text
evaluation/failure_analysis.csv
```

## Main Failure Pattern

Failure type:

```text
Partial body / alignment risk
```

Root cause:

The original body-frame validation required lower-body landmarks including knees and ankles, even when the selected garment was an upper-body garment.

This caused otherwise usable upper-body images to be rejected.

Severity:

```text
Medium
```

Possible fix:

Use garment-category-aware landmark validation so upper-body garments require only relevant upper-body landmarks.

---

# 7. Pipeline Improvements

Two pipeline improvements were implemented.

## Improvement 1 — Category-Aware Landmark Validation

### Before

Upper-body try-on required:

```text
Shoulders
Hips
Knees
Ankles
```

If lower-body landmarks were unavailable, the image was rejected.

### After

Upper-body garments now require:

```text
Left shoulder
Right shoulder
Left hip
Right hip
```

Lower-body landmarks are required only for lower-body or full-body garment categories.

This prevents unnecessary rejection of valid upper-body try-on inputs.

---

## Improvement 2 — Boundary-Tolerant Landmark Validation

### Before

Landmark coordinates had to satisfy:

```text
0.0 <= x <= 1.0
0.0 <= y <= 1.0
```

This could reject useful landmarks close to the frame edge.

### After

A controlled tolerance is allowed:

```text
-0.05 <= x <= 1.05
-0.05 <= y <= 1.05
```

This reduces false rejections while keeping validation bounded.

---

# 8. Evaluation After Improvements

After applying the two improvements, the exact same 24 test combinations were rerun.

Results:

```text
Tests run: 24
Successful: 24
Failed: 0
Failure rate: 0.00%

Average inference time: 0.9014 seconds
Average total processing time: 3.2924 seconds
```

---

# 9. Before vs After Comparison

| Metric | Before | After |
|---|---:|---:|
| Tests Run | 24 | 24 |
| Successful | 16 | 24 |
| Failed | 8 | 0 |
| Failure Rate | 33.33% | 0.00% |
| Average Inference Time | 1.2138 s | 0.9014 s |
| Average Total Processing Time | 4.1152 s | 3.2924 s |

## Key Result

The quality-validation improvements eliminated false `PARTIAL_BODY` rejections for the current upper-body garment evaluation set.

The main improvement came from better input-validation logic rather than increasing model complexity.

---

# 10. Existing Visual Quality Evaluation

Previous structured human evaluation produced:

| Metric | Average Score |
|---|---:|
| Garment Alignment | 3.67 / 5 |
| Garment Preservation | 4.50 / 5 |
| Body Alignment | 3.50 / 5 |
| Boundary Quality | 3.50 / 5 |
| Face Preservation | 5.00 / 5 |
| Overall Realism | 3.50 / 5 |

Overall average:

```text
3.94 / 5
```

The weakest previously observed cases were:

- pair_001
- pair_003
- pair_005

Primary limitations:

- Pose-based garment alignment
- Sleeve/arm occlusion
- Segmentation/compositing boundaries
- Geometric garment deformation
- Baseline model realism

---

# 11. Current Strengths

The current prototype demonstrates:

- Automated batch evaluation
- Quality gating before inference
- Pose quality measurement
- Mask quality measurement
- Garment validation
- Controlled robustness testing
- Automated failure classification
- Before/after comparison
- Processing-time measurement
- Repeatable evaluation using metadata
- Stable result generation for the current evaluation set

---

# 12. Current Limitations

The current system remains a baseline pose-aware 2D try-on prototype.

Known limitations include:

- It is not a photorealistic generative VTON system
- Cloth deformation is geometric
- Arm/garment occlusion handling is limited
- Complex poses may reduce alignment quality
- Current authorized garment set primarily covers upper-body garments
- Dress and trouser evaluation requires additional authorized test images
- Automated alignment indicators do not replace human visual evaluation
- Production deployment requires additional model, security, licensing, and performance review

---

# 13. Files Generated

```text
evaluation/
├── persons/
├── garments/
├── expected/
├── generated/
├── metadata.csv
├── evaluation_results.csv
├── evaluation_results_before.csv
├── failure_analysis.csv
├── failure_analysis_before.csv
├── create_metadata.py
├── run_evaluation.py
└── analyze_failures.py
```

Project-level report:

```text
MODEL_EVALUATION_REPORT.md
```

---

# 14. Final Evaluation Status

The current Raritone VTON prototype now supports:

```text
Input
↓
Quality Validation
↓
Batch Evaluation
↓
Try-On Inference
↓
Automated Metrics
↓
Failure Analysis
↓
Pipeline Improvement
↓
Before/After Comparison
↓
Evaluation Report
```

Final current evaluation status:

```text
24 evaluation combinations
24 successful after improvements
0% pipeline failure rate
0.9014 s average inference time
3.2924 s average total processing time
```