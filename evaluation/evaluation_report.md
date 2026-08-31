\# Image-to-3D Evaluation Report



\## Objective



Evaluate whether a product image can be converted into a usable 3D product asset using GPU inference.



\## Environment



\- Platform: Kaggle

\- GPU: NVIDIA Tesla T4

\- VRAM: 14.56 GB

\- CUDA: Available

\- Model: TripoSR

\- Commit: `107cefdc244c39106fa830359024f6a2f1c78871`



\## Experiment 1 — T-Shirt



The T-shirt product image successfully passed through:



\- Image preprocessing

\- GPU inference

\- Mesh extraction

\- OBJ export



\### Timing



| Stage | Time |

|---|---:|

| Model initialization | 6.388 s |

| Image preprocessing | 0.043 s |

| Model inference | 1.096 s |

| Mesh extraction | 2.232 s |



\### Output



Generated mesh:



`output/0/mesh.obj`



\## Texture Baking Issue



Texture baking was attempted after successful mesh extraction.



The process failed while ModernGL tried to create an X11 rendering context in Kaggle's headless environment.



Observed error:



`XOpenDisplay: cannot open display`



This is a rendering-environment issue and not an image-to-3D inference failure.



\## GLB Export



The post-processing pipeline includes conversion from OBJ to GLB using `trimesh`.



The GLB validation stage is designed to record:



\- Vertex count

\- Polygon count

\- File size

\- Bounding dimensions

\- Watertight status

\- Output format



\## Quality Review



A numerical visual-quality score has not been assigned yet because the generated 3D model still requires manual visual inspection.



\## Additional Products



The following products are included as planned experiment cases:



\- Shirt

\- Shoe

\- Handbag

\- Dress



These are marked as `not\_run` and are not reported as completed experiments.



\## Conclusion



TripoSR was successfully configured and executed on a Kaggle Tesla T4 GPU.



A T-shirt product image was successfully converted into a 3D mesh.



Current asset status:



`PENDING\_REVIEW`

