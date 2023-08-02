# SSKF and MultiClass-ASKFSVM

This repository contains the prototype code for Static and Adaptive Subspace Kernel Fusion SVM.

Heterogeneous data is ubiquitous in various applications, from healthcare to environmental sciences, and presents as diverse formats such as text, images, and video. Traditional machine learning algorithms often struggle to effectively analyze this array of data types.

Our work introduces two novel approaches: **Static Subspace Kernel Fusion** and **Adaptive Subspace Kernel Fusion**. The **Static approach** is a kernel-based method that extracts essential components from each input modality's subspace, creating a unified data representation. The **Adaptive approach** introduces an adaptation step. By integrating the weighting of spectral properties into the fusion process, we aim to enhance the data representation for specific classification tasks.

## Prerequisites

This project is optimized to run with Python 3.10.
Fetch all dependencies for SSKF-SVM and ASKF-SVM by

```console
pip install -r requirements.txt
```

## Work in progress

SSKF-SVM and ASKF-SVM are under development that means that not all features will work perfectly fine.
Currently, we are implementing further multi-class strategies as the current state only includes one-vs-rest.
Furthermore, currently, we try to improve speed for multi-class ASKF-SVM by parallelization.

## Citing SSKF-SVM and ASKF-SVM

If you use SSKF-SVM or ASKF-SVM for a scientific purpose, please cite the following paper:

```
@article{MUNCH2023126635,
  title = {Static and adaptive subspace information fusion for indefinite heterogeneous proximity data},
  journal = {Neurocomputing},
  pages = {126635},
  year = {2023},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231223007580},
  author = {Maximilian Münch and Manuel Röder and Simon Heilig and Christoph Raab and Frank-Michael Schleif},
}
```



