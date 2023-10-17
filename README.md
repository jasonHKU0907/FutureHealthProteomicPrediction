<div align="center">

![Logo](./src/Logo.png?raw=true "Logo")

</div>

## Description   
Code related to the paper "Plasma proteomic profiles predict individual future health risk". 
This repository contains python codes for data preporocessing, model training and evaluations of the proposed Proteomic Neural Network.

![Workflow](./src/Study_Flowchart.png?raw=true "Workflow")

## Methods
The **Proteomic Neural Network** was developed based on [Keras](https://github.com/keras-team/keras). The ProNNet served as a feature extractor to translate the proteomic data into a future incident risk probabilities corresponding to 45 endpoints, covering different categories of diseases and mortalities.

![Architecture](./src/ProNNet.png?raw=true "Architecture")

## Assets
This repo contains code to preprocess [UK Biobank](https://www.ukbiobank.ac.uk/) data, train the ProNNet and analyze/evaluate its performance.

- **DataPreparation**
- Involves target data generation for different endpoints. Proteomic data and Clinical data (PANEL) generation and pre-processing.
- **ModelTraining**
- Proteomic neural network development and downstream survival analysis of Cox proportional hazard regressions.
- **Figures**
- Code to generate figures in the mauscripts.
- **Tables**
- Code to generate supplementary tables in the mauscripts.
- **Utility**
- Implement function calls for training and evaluation procedures.

  
## License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].
[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]


## Citation   
```
@article{buergel2022metabolomic,
  title={Plasma proteomic profiles predict individual future health risk},
  author={You, Jia and Guo, Yu and Zhang, Yi and Kang, Ju-Jiao and Wang, Lin-Bo and Feng, Jian-Feng and Cheng, Wei and Yu, Jin-Tai},
  Journal information will update later
}
```

