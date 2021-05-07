# DeepRecS-From-RECIST-Diameters-to-Precise-Liver-Tumor-Segmentation
**Requirements for RECIST Mark Propagation:**
- python==2.7.15
- Keras==2.1.6
- Tensorflow-gpu==1.12.0

Training data: **cropped tumor patches**+**GT RECIST points** (row, colomn coordinates of four endpoints)

**Requirements for Liver Tumor Segmentation:**
- python==3.6.8
- Keras==2.2.4
- Tensorflow-gpu==1.12.0

Training data: **cropped tumor patches**+**RECIST images**+**GT tumor delineations**+**GT tumor boundaries**
