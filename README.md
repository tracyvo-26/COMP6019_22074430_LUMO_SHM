# COMP6019 Code

```
code_clean/
├── data/                         Raw LUMO data
├── 01_data_and_features/         Feature extraction and feature analysis
│   ├── between channel and temp explore/
│   ├── feature_distributions/
│   ├── figures_and_tables/
│   └── spatial_features/
├── 02_modal_identification/      FDD, SSI-COV and MAC validation
│   ├── FDD/
│   ├── SSI-COV/
│   └── MAC/
├── 03_models/                    Graph construction and GIN / GIN+CAL models
│   ├── graph_builder.py
│   ├── cal_original/             Original CAL code (Sui et al.)
│   └── gin_cal/
│       ├── core/                 Shared training and evaluation code
│       ├── feature_assessment/   Feature assessment and reference predictors
│       ├── settings_1_original_cal/
│       ├── settings_2_project_gin/
│       ├── diagnosis/
│       ├── tuning/
│       ├── exploratory/
│       └── results/              CSV results
├── 04_sensor_selection/          Sensor placement
│   ├── classical_osp/
│   └── ablation/
├── 05_fe_model/                  Finite element model and parser
├── data_checks/                  Early checks of the raw data
```
