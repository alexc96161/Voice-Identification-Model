This project trains an AI voice identification model to detect if a voice is my own or someone else's.
It implements scripts to acquire sentences to be recorded, preprocess wave files into MFCC json files, and train a model with Convolutional Layers using Tensorflow. 
The best model after running the training script will be stored in checkpoints as a keras file. Uses the built-in multiprocessing library for faster preprocessing.


The current best model (.keras) has ~97.5% validation test accuracy. The older best model (.hdf5) has a lower validation test accuracy (~96%), but has significantly fewer model weights.


To install prerequisite libraries, use ``pip install -r requirements.txt``. Tested using python 3.12.4.


Notes:  
The current model could not be uploaded due to the github 100MB limit. You can generate a new best model by running VoiceReg.py  
This code was updated to use python 3.12 and assosiated libraries. The old best model was from multiple library/python versions ago.
