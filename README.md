# Changelog
- 04/12/2021 Anemia images were added to the "RBC Diseases" folder.
- 24/11/2021 Elliptocyte was added to the dataset with 32 blood smear images and labels.

# Chula RBC-12-Dataset

Chula-RBC-12 Dataset is a dataset of red blood cell (RBC) blood smear images used in "**Red Blood Cell Segmentation with Overlapping Cell Separation and Classification from an Imbalanced Dataset**", containing 12 classes of RBC types consisting of 706 smear images that contain over 20,000 RBC cells. The dataset was collected at the Oxidation in Red Cell Disorders Research Unit, Chulalongkorn University in 2019 using a DS-Fi2-L3 Nikon microscope at 1000x magnification.

## RBC Types
- 0 Normal cell
- 1 Macrocyte
- 2 Microcyte
- 3 Spherocyte
- 4 Target cell
- 5 Stomatocyte
- 6 Ovalocyte
- 7 Teardrop
- 8 Burr cell
- 9 Schistocyte
- 10 uncategorised
- 11 Hypochromia
- 12 Elliptocyte



## Dataset
- The "Dataset" folder contains 738 RBC blood smear images with 640*480 resolution.
- The "Label" folder contains label data with the name of the corresponding image. The file can contain multiple line. Each line is stored in the following sequence:
- x coordinate
- y coordinate
- type of RBC in number



## Citation
If you use this dataset, please cite the following paper:

```
@misc{naruenatthanaset2021red,
      title={Red Blood Cell Segmentation with Overlapping Cell Separation and Classification on Imbalanced Dataset}, 
      author={Korranat Naruenatthanaset and Thanarat H. Chalidabhongse and Duangdao Palasuwan and Nantheera Anantrasirichai and Attakorn Palasuwan},
      year={2021},
      eprint={2012.01321},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
