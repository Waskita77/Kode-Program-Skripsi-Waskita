# Thesis Code Archive: Transfer Ensemble Learning for Large-Scale LULC Classification

This repository contains the original Python scripts used in my undergraduate thesis:

**"Penerapan Transfer Ensemble Learning untuk Evaluasi Metode Machine Learning dalam Klasifikasi Penutup dan Penggunaan Lahan Skala Besar"**

Published thesis record:  
https://etd.repository.ugm.ac.id/penelitian/detail/269359

## Project Overview

This research evaluates machine learning and stacking ensemble models for large-scale land cover and land use classification using PlanetScope imagery.

The study uses:

- PlanetScope Dove-R 2019 as the source domain
- PlanetScope SuperDove 2024 as the target domain
- Spectral bands, NDVI, and GLCM-derived texture features
- Single machine learning models and stacking ensemble models

The main objective of this research is to evaluate the performance and transferability of machine learning and ensemble learning models under temporal and sensor domain shift.

## Tested Models

| Code | Model |
| --- | --- |
| T-1 | Support Vector Machine with Linear kernel |
| T-2 | Support Vector Machine with RBF kernel |
| T-3 | Random Forest |
| T-4 | XGBoost |
| E-1 | Stacking ensemble of SVM-based models |
| E-2 | Stacking ensemble of tree-based models |
| E-3 | Stacking ensemble of all base models |

## Workflow

The general workflow consists of:

1. Satellite imagery preparation
2. NDVI generation
3. GLCM texture feature extraction
4. Raster stacking and sample extraction
5. Dataset cleaning and train-test splitting
6. Source-domain model training
7. Baseline evaluation on the source domain
8. Zero-shot transfer evaluation on the target domain
9. Fine-tuning using target-domain samples
10. Raster-based inference

Some geospatial preprocessing steps, such as NDVI generation, raster stacking, rasterization, and sample extraction, were conducted using ArcGIS Pro and QGIS. The Python scripts in this repository focus mainly on model training, evaluation, and raster inference.

## Repository Contents

| File | Description |
| --- | --- |
| `0.preparation.ipynb` | Notebook for dataset preparation, cleaning, and train-test splitting |
| `1.1.model_train.py` | Source-domain model training |
| `1.2.model_train(fine-tune).py` | Target-domain fine-tuning |
| `2.1model_evaluation.py` | Baseline and zero-shot evaluation |
| `2.2.model_evaluation(fine-tune).py` | Fine-tuned model evaluation |
| `3.1.model_inference.py` | Raster inference using source-domain models |
| `3.2.model_inference(fine-tune).py` | Raster inference using fine-tuned models |

## Notes

This repository contains the original code archive used for the implementation of my undergraduate thesis.

The original PlanetScope imagery, government-provided land-cover and land-use reference data, intermediate raster products, sampled CSV datasets, trained model files, and generated classification maps are not included in this repository due to file size, data access restrictions, and redistribution limitations.

The ground truth data used in this research was obtained from government-provided land-cover and land-use data and is therefore not publicly redistributed through this repository.

Several geospatial preprocessing steps, including NDVI generation, raster stacking, rasterization, and sample extraction, were conducted using ArcGIS Pro and QGIS before the Python-based machine learning workflow.

As a result, this repository is maintained as a documented thesis code archive rather than a fully reproducible end-to-end software package.

## Author

**Waskita Abdillah Rafiqi**  
Cartography and Remote Sensing Study Program  
Department of Geographic Information Science  
Faculty of Geography, Universitas Gadjah Mada
