## Training
follow run_train_sdreamer.py

To prepare for the training dataset, follow write_data_sdreamer.py

## Citing sDREAMER
Please cite [the paper below](https://www.cs.rochester.edu/u/yyao39/files/sDREAMER.pdf) when you use sDREAMER
```
@INPROCEEDINGS{10224751,
  author={Chen, Jingyuan and Yao, Yuan and Anderson, Mie and Hauglund, Natalie and Kjaerby, Celia and Untiet, Verena and Nedergaard, Maiken and Luo, Jiebo},
  booktitle={2023 IEEE International Conference on Digital Health (ICDH)}, 
  title={sDREAMER: Self-distilled Mixture-of-Modality-Experts Transformer for Automatic Sleep Staging}, 
  year={2023},
  volume={},
  number={},
  pages={131-142},
  keywords={Training;Sleep;Brain modeling;Transformers;Electromyography;Electroencephalography;Electronic healthcare;sleep scoring;distillation;transformer;mixture-of-modality experts},
  doi={10.1109/ICDH60066.2023.00028}}
```

# Original README below
# Flow

# File Structure
```bash
├── README.md
├── data
│   ├── dst_data --> processed data wo NE (1st version)
│   ├── dst_data_wNE --> processed data with NE(2nd version)
│   ├── raw_data --> raw data (1st version)
│   └── raw_data_wNE --> raw data with NE (2nd version)
├── ckpt 
│   ├── baseline --> ckpts for baseline model under different settings
│   ├── cm
│   ├── ...
│   └── ...
├── ckpt_ne
│   ├── baseline --> ckpts for baseline model under different settings
│   └── sdreamer 
├── ckpt_seq --> used to store ckpts for seq2seq model(no longer used)
├── data_provider
│   ├── __init__.py
│   ├── data_generator.py --> used to create data and data loader object
│   ├── data_generator_ne.py --> used to create data and data loader object for NE
│   └── data_loader.py --> used to load and preprocess data
├── epoch_pics --> manually created folder to store demo timeseries pics(no longer used)
├── exp
│   ├── exp_main.py --> main file to run experiments for non-MoE models
│   ├── exp_moe_ne.py --> main file to run experiments for MoE models with NE(not used yet)
│   ├── exp_moe.py --> main file to run experiments for MoE models(no longer used)
│   ├── exp_moe2.py --> main file to run experiments for MoE models(currently used)
│   └── exp_ne.py --> main file to run experiments for non-MoE models with NE
├── layers --> all sub-modules used in the model
├── logs --> all training logs are stored here
├── models
│   ├── epoch --> all epoch models (NE models included)
│   └── seq --> all seq models (NE models included)
├── scripts
│   ├── epoch --> used to run all non-NE epoch experiments(subfolders specified by each model)
│   └── seq --> used to run all non-NE seq experiments
├── scripts_ne
│   ├── epoch --> used to run all NE epoch experiments(subfolders specified by each model)
│   └── seq --> used to run all NE seq experiments
├── utils --> some utility functions used in the project
├── visualizations --> visualized results are stored here
├── ml_baseline.ipynb --> baseline ML models
├── moe_Eval.py --> used to evaluate MoE models
├── moe_Launch.py --> used to train MoE models(no longer used)
├── moe_Launch2.py --> used to train MoE models(currently used)
├── moe_LaunchNE.py --> used to train MoE models with NE(not used yet)
├── train_Launch.py --> used to train non-MoE models wo NE(1st version)
└── train_LaunchNE.py --> used to train non-MoE models with NE(2nd version)
```