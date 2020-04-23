# Music_Genre_Classifier

This is a classifier trained using two different neural network models: Multi-Layer Perceptron(MLP) and Convolutional Neural Network(CNN).

Marsyas Audio file library is used for training the model.

Before building the models, JSON file dataset is written by doing preprocessing on the audio files and extracting the MFCCs for the data. This preprocessing is done using the librosa library

MLP and CNN architecture is built using the Keras neural network library in the TensorFlow library.

Dropout and Regularization are also done in both the models to avoid the problem of overfitting.
