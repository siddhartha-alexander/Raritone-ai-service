\# Raritone Image-to-3D Pipeline



\## Overview



This pipeline converts a product image into a 3D product asset using GPU-based image-to-3D reconstruction.



\## Architecture



Product Image  

↓  

Kaggle GPU  

↓  

TripoSR  

↓  

3D Mesh Extraction  

↓  

OBJ  

↓  

GLB Export  

↓  

Validation  

↓  

Metadata \& Evaluation  

↓  

PENDING\_REVIEW



\## Image-to-3D Model



Model: TripoSR



Repository:

VAST-AI-Research/TripoSR



Pinned commit:



`107cefdc244c39106fa830359024f6a2f1c78871`



The model version is pinned to improve reproducibility.



\## GPU Environment



Testing environment:



\- Platform: Kaggle

\- GPU: NVIDIA Tesla T4

\- VRAM: 14.56 GB

\- CUDA: Enabled

\- PyTorch: 2.10.0+cu128



\## Generation



The product image is provided to TripoSR.



The model performs:



1\. Image preprocessing

2\. GPU inference

3\. 3D representation generation

4\. Mesh extraction

5\. Mesh export



\## Verified Result



The T-shirt experiment successfully generated:



`mesh.obj`



Measured performance:



\- Model initialization: 6.388 s

\- Image processing: 0.043 s

\- Model inference: 1.096 s

\- Mesh extraction: 2.232 s



\## GLB Export



The post-processing stage uses `trimesh` to convert the generated OBJ mesh into GLB.



GLB is the target format for downstream browser-based 3D visualization.



\## Validation



The validation stage is designed to inspect:



\- Vertex count

\- Polygon count

\- Bounding box

\- Dimensions

\- File size

\- Watertight status

\- Output format



\## Texture Generation



Texture baking was attempted.



Mesh generation completed successfully, but ModernGL failed to create an X11 context inside Kaggle's headless runtime.



Error:



`XOpenDisplay: cannot open display`



A future version can use an EGL-compatible headless rendering backend for texture baking.



\## Metadata



Generated assets should contain:



\- Asset ID

\- Product ID

\- Category

\- Model

\- Model version/commit

\- GPU

\- Output format

\- Geometry statistics

\- Input provenance

\- Review status



\## Provenance



Production assets must retain the actual source and license of the input product image.



Input provenance must not be inferred.



\## Review State



All newly generated assets enter:



`PENDING\_REVIEW`



before being approved for downstream Raritone usage.

