#----RICE LEAF DISEASE DATASET ----

this directory contains the image data specificaiton for training and evaluating the 'Rice-DOctpe' MobileNetV4 architecture.

##--Target Disease Clasess--

Our model performs 4-class classification on the following leaf disease condition:

|Class name | Pathagen / Cause | Visual symptoms on leaf|
| :----- | :----- | :------ |
| **Bacterial_Leaf_Blight** | *Xanthomonas oryzae* (Bacterium) | Long yellowish-white wavy stripes starting from leaf tips and margins |
| **Rice_Blast** | *Pyricularia oryzae* (Fungus) | Spindle/diamond-shaped lesions with dark brown borders and gray centers |
| **Brown_Spot** | *Bipolaris oryzae* (Fungus) | Small circular or oval dark brown spots across the leaf blade |
| **Healthy** | Normal leaf (Negative control) | Uniform green leaf tissue with no lesions or yellowing |



##--Required Directory Structure--

Organize the extracted images into `train/` and `val/` subfolders matching these exact names:

```text
data/
└── rice_leaf_diseases/
    ├── train/
    │   ├── Bacterial_Leaf_Blight/
    │   ├── Brown_Spot/
    │   ├── Healthy/
    │   └── Rice_Blast/
    └── val/
        ├── Bacterial_Leaf_Blight/
        ├── Brown_Spot/
        ├── Healthy/
        └── Rice_Blast/



##--DATASET SOURCE DOWNLOAD--

The base training images are sourced from public foliar pathology benchmarks:
- **Kaggle Dataset:** ______________.

To download and extract directly via the Kaggle CLI:
```bash
kaggle datasets download -d <dataset-name-here>
unzip __________.zip -d data/rice_leaf_diseases/