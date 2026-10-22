# Dataset Sources and Reproducibility

## UCI HAR

Dataset: Human Activity Recognition Using Smartphones

Official source:
https://archive.ics.uci.edu/dataset/240

Dataset DOI:
https://doi.org/10.24432/C54S4K

License: CC BY 4.0

Citation:
Reyes-Ortiz, J., Anguita, D., Ghio, A.,
Oneto, L., and Parra, X. (2013).
Human Activity Recognition Using Smartphones.
UCI Machine Learning Repository.

Preparation:
1. Download the official dataset.
2. Extract it under:
   data/uci_har/UCI HAR Dataset/
3. Run:
   python examples/prepare_uci_har.py

Development experiments use only:
out/uci_har/uci_har_official_train.csv

The official test partition must not be used
for development partition searches.

Expected development data:
- 7,352 records
- 21 subjects
- 6 classes
- 561 features

## MHEALTH

Dataset: MHEALTH

Official source:
https://archive.ics.uci.edu/dataset/319

Dataset DOI:
https://doi.org/10.24432/C5TW22

License: CC BY 4.0

Citation:
Banos, O., Garcia, R., and Saez, A. (2014).
MHEALTH. UCI Machine Learning Repository.

The UCI dataset page additionally requests
acknowledgement of the associated mHealthDroid
and mobile-health framework publications.

Preparation:
1. Download the official MHEALTH dataset.
2. Extract all ten subject logs under:
   data/mhealth/
3. Run:
   python examples/prepare_mhealth.py

Expected prepared data:
- 1,215,745 records
- 10 subjects
- 23 sensor features
- Labels 0 through 12
- Background class 0 retained

## Dataset Integrity

Dataset provenance metadata and original-source
SHA-256 fingerprints are recorded in:

reproducibility/dataset_provenance.json

Use this record to compare downloaded source
files and generated datasets.

## Redistribution and Attribution

SplitShiftLab software is licensed under MIT.

The UCI HAR and MHEALTH datasets are separately
licensed under CC BY 4.0 and require attribution.

The research repository does not include
redistributed raw datasets in its current release.

Users should obtain datasets from the official
sources and follow the preparation instructions.
