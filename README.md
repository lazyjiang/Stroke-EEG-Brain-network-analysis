# Stroke-EEG-Brain-network-analysis
Functional connectivity and brain network (graph theory) analysis for motor imagery data of stroke patiens.

## Installation
1. Create a conda environment
```bash
conda create -n env_name python=3.10
```
2. Install pip package
```bash
pip install -r requirements.txt
```
3. Install seaborn package
```bash
pip install seaborn==0.12.0
```

## Dataset
The EEG dataset of stroke patients is provided by Liu et.al in https://doi.org/10.6084/m9.figshare.21679035.v5  
You just need to download "sourcedata.zip" through this link and unzip it to the "dataset/sourcedata" directory.


## Usage
1. Python file: figshare_stroke_fc2.py
- Plot functional connectivity matrix and corresponding topology in 3 frequency bands for 50 stroke patients. 
- Save the functional connectivity data (imcoh_left.npy and imcoh_right.npy) to data_load/ImCoh_data.
```bash
python figshare_stroke_fc2.py
```
2. Python file: figshare_fc_mst2.py
- Calculate and visualize the maximum spanning tree (MST) transformed from the function connectivity matrix.
- Correlation analysis: regplot between the NIHSS score and various MST metrics (diameter, eccentricity, leaf number, tree hierarchy).
- Comparision analysis: violinplot of the MST metrics under the low NIHSS group and high NIHSS group.
- Correct the correlation coefficient by Spearman correlation and permutation test.
```bash
python figshare_fc_mst2.py
```

## Directory
```
- dataset/
  - sourecedata/
    - sub-01/
    - sub-02/
    ...
    - sub-50/
  - subject.csv
- data_load/
  - ImCoh_data/
    - alpha_beta12/
      - imcoh_left.npy
      - imcoh_right.npy
- figshare_stroke_fc2.py
- figshare_fc_mst2.py
```

