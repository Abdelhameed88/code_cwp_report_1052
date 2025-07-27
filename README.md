# Reproducible Code for CWP Report 1052
This repository contains the code used to produce the results of **CWP Report 1052** (May 2025) titled: Efficient Multiparameter Seismic Monitoring using Wavelet1 Transforms and Machine Learning: A Double-Compression Approach

## Contact Information

- **Name**: Ahmed Ahmed  
- **Email**: ahmedmohamedahmed@mines.edu

## Dependencies

This project requires the following key Python packages:

- Python 3.8+
- PyTorch >= 1.10
- torchvision
- numpy, scipy, matplotlib
- scikit-learn
- tensorboard
- Custom modules: `utils.py`, `network.py`, `transforms.py`, `dataset.py`, `scheduler.py`, `vis.py`

## Project Structure and Steps

### 1. Preprocessing (`01_Preprocessing`)
- Includes Jupyter notebooks for model detrending and data preparation.

### 2. Wavelet Transform for Model Compression and Reconstruction (`02_model_compression_reconstruction`)
- Jupyter notebooks are provided to test five wavelet basis functions:
  - Haar
  - Daubechies 2 (DB2)
  - Symlets 2 (Sym2)
  - Coiflets 2 (Coif2)
  - Biorthogonal

### 3. Training Multiresolution Neural Networks (`03_Multiresolution_network`)
- Two training configurations are provided:
  
  **a. `2Encoder_4Decoder`**  
  - Trains a neural network on four models reconstructed from levels 0 to 3 of the wavelet decomposition.

  **b. `2Encoder_2Decoder`**  
  - Trains a neural network on two models reconstructed from levels 4 and 5.

## Training and Testing Instructions

### To Train a Model from Scratch
```bash
python train_mod1.py -ds kimberlina -n YOUR_DIRECTORY -m YOUR_NETWORK --tensorboard -t train.txt -v val_noiseFree.txt

python test.py -ds kimberlina -n YOUR_DIRECTORY -m YOUR_NETWORK -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3

