import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
import pickle

from tensorflow.data import Dataset
from tensorflow.keras.backend import clear_session

PREDICT_BATCH_SIZE = 2**20
GPU_ID = -1

from tensorflow import config as tfconfig

gpus = tfconfig.list_physical_devices('GPU')
if gpus:
    try:
        tfconfig.set_visible_devices(gpus[GPU_ID], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tfconfig.experimental.set_memory_growth(gpu, True)
        logical_gpus = tfconfig.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        raise RuntimeError(e) # for now just raise the error and exit
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Data generator
class DataGenerator(Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_data = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        return batch_data

class NeuralNetwork:
    """
    This class implements a neural network classifier.

    Attributes:
        model (Sequential): The neural network model.
        scaler (StandardScaler): The scaler used for feature scaling.

    Methods:
        __init__(self, train_data): Initializes the NeuralNetwork object.
        fit(self, train_data, y_train, weights_train=None): Fits the neural network model to the training data.
        predict(self, test_data): Predicts the output labels for the test data.
        save(self, model_name): Saves the trained model and scaler to disk.
        load(self, model_path): Loads a trained model and scaler from disk.

    """

    def __init__(self, n_dim=None, train_data=None):
        self.model = Sequential()

        if n_dim is None:
            n_dim = train_data.shape[1]

        self.model.add(Dense(100, input_dim=n_dim, activation="relu"))
        self.model.add(Dense(100, activation="relu"))
        self.model.add(Dense(100, activation="relu"))        
        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.scaler = StandardScaler()

    def fit(self, train_data, y_train, weights_train=None):
        """
        Fits the neural network model to the training data.

        Args:
            train_data (pandas.DataFrame): The input training data.
            y_train (numpy.ndarray): The target training labels.
            weights_train (numpy.ndarray, optional): The sample weights for training data.

        Returns:
            None

        """
        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        self.model.fit(X_train, y_train, sample_weight=weights_train, epochs=2, verbose=2)

    def predict(self, test_data, batch_size=PREDICT_BATCH_SIZE, verbose='auto'):
        """
        Predicts the output labels for the test data.

        Args:
            test_data (pandas.DataFrame): The input test data.

        Returns:
            numpy.ndarray: The predicted output labels.

        """
        if verbose != 'auto' and verbose > 0:
            print('test_data.shape in NeuralNetwork.predict', test_data.shape)
            print('type(test_data)', type(test_data))
        test_data = self.scaler.transform(test_data)
        if verbose != 'auto' and verbose > 0:
            print('type(test_data) after transform', type(test_data))
        # test_data = Dataset.from_tensor(test_data)

        # Create data generator
        data_gen = DataGenerator(test_data, PREDICT_BATCH_SIZE)

        # from IPython import embed;embed()  # fmt: skip
        result = self.model.predict(data_gen, verbose=verbose).flatten().ravel() # test_data, batch_size=batch_size
        if verbose != 'auto' and verbose > 0:
            print(f'Done. result.shape: {result.shape}')
        return result
    
    def save(self, model_name):
        """
        Saves the trained model and scaler to disk.

        Args:
            model_name (str): The name of the model file to be saved.

        Returns:
            None

        """
        model_path = model_name + ".keras"
        self.model.save(model_path)
        
        scaler_path = model_name + ".pkl"
        pickle.dump(self.scaler, open(scaler_path, "wb"))
        
    def load(self, model_path):
        """
        Loads a trained model and scaler from disk.

        Args:
            model_path (str): The path to the saved model file.

        Returns:
            Sequential: The loaded model.

        """
        self.model = load_model(model_path)
        self.scaler = pickle.load(open(model_path.replace(".keras", ".pkl"), "rb"))
        
        return self.model
