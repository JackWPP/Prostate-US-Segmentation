# Micro-Ultrasound Prostate Segmentation Dataset

### Creators

* [Shao, Wei (Contact person)^1^](https://zenodo.org/search?q=metadata.creators.person_or_org.name:%22Shao,+Wei%22)[![ORCID icon](https://zenodo.org/static/images/orcid.svg)](https://orcid.org/0000-0003-4931-4839 "Shao, Wei's ORCID profile")
* [Brisbane, Wayne (Data collector)^2^](https://zenodo.org/search?q=metadata.creators.person_or_org.name:%22Brisbane,+Wayne%22)[![ORCID icon](https://zenodo.org/static/images/orcid.svg)](https://orcid.org/0000-0003-0470-5262 "Brisbane, Wayne's ORCID profile")

## Description

This dataset comprises micro-ultrasound scans and human prostate annotations of 75 patients who underwent micro-ultrasound guided prostate biopsy at the University of Florida. All images and segmentations have been fully de-identified in the NIFTI format.

Under the "train" folder, you'll find three subfolders:

1. "micro_ultrasound_scans" contains micro-ultrasound images from 55 patients for training.
2. "expert_annotations" contains ground truth prostate segmentations annotated by our expert urologist.
3. "non_expert_annotations" contains prostate segmentations annotated by a graduate student.

In the "test" folder, there are five subfolders:

1. "micro_ultrasound_scans" contains micro-ultrasound images from 20 patients for testing.
2. "expert_annotations" contains ground truth prostate segmentations by the expert urologist.
3. "master_student_annotations" contains segmentations by a master's student.
4. "medical_student_annotations" contains segmentations by a medical student.
5. "clinician_annotations" contains segmentations by a urologist with limited experience in reading micro-ultrasound images.

If you use this dataset, please cite our paper: Jiang, Hongxu, et al. "MicroSegNet: A deep learning approach for prostate segmentation on micro-ultrasound images." Computerized Medical Imaging and Graphics (2024): 102326. DOI: https://doi.org/10.1016/j.compmedimag.2024.102326.

For any dataset-related queries, please reach out to Dr. Wei Shao:  [weishao@ufl.edu]().
