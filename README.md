# Reproducble code for cwp report 1052
The code used to produce the results of CWP report 1052 - May 2025

## Contact information:

- Name: Ahmed Ahmed

- email: ahmedmohamedahmed@mines.edu

```markdown
## Dependencies

This project requires the following key packages:

- Python
- PyTorch >= 1.10
- torchvision
- numpy, scipy, matplotlib
- scikit-learn
- tensorboard
- Custom modules: `utils.py`, `network.py`, `transforms.py`, `dataset.py`, `scheduler.py`, `vis.py`

The project contains three main steps:

- Preprocessing step (01_Preprocessing): provided Jupyter notebooks for model detrending

- Wavelet transform for model compression and reconstruction (02_model_compression_reconstruction): provided Jupyter notebook for the testing of five wavelet basis functions (Haar, DB2, Sym2, Coif2, Biorth)

- Training multirsolution neural networks (03_Multiresolution_network): two directories provided

        1- "2Encoder_4Decoder" includes the required codes to train a neural network for four models simultaneously from the model reconstructed with base level up to the model reconstructed with three levels of decomposition.
        2- "2Encoder_2Decoder" includes the required codes to train a neural network for two models simultaneously, the model reconstructed with four and five levels of decomposition.

To start train a network from scratch:

python train_mod1.py -ds kimberlina -n YOUR_DIRECTORY -m YOUR_NETWORK --tensorboard -t train.txt -v val_noiseFree.txt

To test the trained model

python test.py -ds kimberlina -n YOUR_DIRECTORY -m YOUR_NETWORK -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3


