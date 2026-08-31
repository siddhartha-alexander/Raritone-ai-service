\# Image-to-3D Product Generation



GPU-based product image-to-3D generation pipeline for Raritone using TripoSR.



\## Pipeline



Product Image → Kaggle Tesla T4 → TripoSR → 3D Mesh → GLB → Validation → Evaluation



\## Model



\- Model: TripoSR

\- GPU: NVIDIA Tesla T4

\- VRAM: 14.56 GB

\- PyTorch: 2.10.0+cu128

\- CUDA: Available

\- Commit: 107cefdc244c39106fa830359024f6a2f1c78871



\## T-Shirt Test



The T-shirt image successfully completed:



\- Image preprocessing

\- GPU inference

\- 3D mesh extraction

\- OBJ export



Output:



`output/0/mesh.obj`



Performance:



\- Model initialization: 6.388 s

\- Image processing: 0.043 s

\- Model inference: 1.096 s

\- Mesh extraction: 2.232 s



\## Texture Baking



Texture baking was tested, but Kaggle's headless environment caused ModernGL to fail while creating an X11 display context.



The 3D mesh generation itself completed successfully.



\## GLB



The generated OBJ can be converted to GLB using trimesh as part of the post-processing pipeline.



\## Status



PENDING\_REVIEW

