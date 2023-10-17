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
This repo contains code to preprocess [UK Biobank](https://www.ukbiobank.ac.uk/) data, train the MetabolomicStateModel and analyze/evaluate its performance.

- Preprocessing involves parsing primary care records for desired diagnosis. 
- Training involves Model specification via pytorch-lightning and hydra.
- Evaluation involves extensive benchmarks with linear Models, and calculation of bootstrapped metrics.
- Visualization contains the code to generate the figures displayed in the paper. 

## License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Citation   
```
@article{buergel2022metabolomic,
  title={Metabolomic profiles predict individual multidisease outcomes},
  author={Buergel, Thore and Steinfeldt, Jakob and Ruyoga, Greg and Pietzner, Maik and Bizzarri, Daniele and Vojinovic, Dina and Upmeier zu Belzen, Julius and Loock, Lukas and Kittner, Paul and Christmann, Lara and others},
  journal={Nature Medicine},
  pages={1--12},
  year={2022},
  publisher={Nature Publishing Group}
}
```

