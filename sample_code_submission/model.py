# ------------------------------
# Dummy Sample Submission
# ------------------------------

NO_PREDICTION = False
STAT_ONLY = True
TEMPLATE_FILE = "templates.npz"

XGBOOST = False
TENSORFLOW = True
TORCH = False

import numpy as np
import os
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split

if STAT_ONLY:
    from stat_only_analysis import StatOnlyAnalysis as StatisticalAnalysis
else:
    from statistical_analysis import  StatisticalAnalysis

current_file = os.path.dirname(os.path.abspath(__file__))


class Model:
    """
    This is a model class to be submitted by the participants in their submission.

    This class consists of the following functions:
    1) __init__:
        Initializes the Model class.
        Args:
            get_train_set (callable, optional): A function that returns a dictionary with data, labels, weights, detailed_labels, and settings.
            systematics (object, optional): A function that can be used to get a dataset with systematics added.
        Returns:
            None

    2) fit:
        Trains the model.
        Params:
            None
        Functionality:
            This function can be used to train a model. If `re_train` is True, it balances the dataset,
            fits the model using the balanced dataset, and saves the model. If `re_train` is False, it
            loads the saved model and calculates the saved information. The saved information is used
            to compute the train results.
        Returns:
            None

    3) predict:
        Predicts the values for the test set.
        Parameters:
            test_set (dict): A dictionary containing the test data and weights.
        Returns:
            dict: A dictionary with the following keys:
            - 'mu_hat': The predicted value of mu.
            - 'delta_mu_hat': The uncertainty in the predicted value of mu.
            - 'p16': The lower bound of the 16th percentile of mu.
            - 'p84': The upper bound of the 84th percentile of mu.

    4) balance_set:
        Balances the training set by equalizing the number of background and signal events.
        Params:
            None
        Returns:
            dict: A dictionary with the balanced training set.
    """

    def __init__(self, get_train_set=None, systematics=None):
        """
        Initializes the Model class.

        Args:
            get_train_set (callable, optional): A function that returns a dictionary with data, labels, weights,detailed_labels and settings.
            systematics (object, optional): A function that can be used to get a dataset with systematics added.

        Returns:
            None
        """
        self.get_train_set = get_train_set
        self.systematics = systematics

        self.counts_dir = os.path.join(
            os.path.dirname(current_file),
            'sample_result_submission',
            input("Enter the directory for the event counts files: "),
        )
        os.makedirs(self.counts_dir, exist_ok=True)

        self.re_train = True



    def fit(self, stat_only: bool = None, syst_settings: dict[str, bool] = None):
        """
        Trains the model.

        Params:
            stat_only (bool, optional): Force to compute stats only results [the highest priority]. Defaults to None.
            syst_settings (dict, optional): Dictionary containing the systematic settings of whether to fix systematics in fitting. For example, {'jes': True}. Defaults to None.

        Functionality:
            This function can be used to train a model. If `re_train` is True, it balances the dataset,
            fits the model using the balanced dataset, and saves the model. If `re_train` is False, it
            loads the saved model and calculates the saved information. The saved information is used
            to compute the train results.

        Returns:
            None
        """
        if os.path.exists(os.path.join(current_file, TEMPLATE_FILE)):
            self.stat_analysis = StatisticalAnalysis(self.model, template_file=os.path.join(current_file, TEMPLATE_FILE))
            return None
        self.train_set = self.get_train_set()
        del self.train_set["settings"]
        self.train_set["labels"] = self.train_set["labels"].astype(np.int8)
        self.train_set["detailed_labels"] = pd.Categorical(
            self.train_set["detailed_labels"],
            ['htautau', 'diboson', 'ttbar', 'ztautau'],
            True,
        )

        # compute derived features. also applies cuts.
        # consider moving this to after the data is split to reduce peak memory use
        self.train_set = self.systematics(self.train_set)

        print("Full data: ", self.train_set["data"].shape)
        print("Full Labels: ", self.train_set["labels"].shape)
        print("Full Weights: ", self.train_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.train_set["weights"][self.train_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.train_set["weights"][self.train_set["labels"] == 0].sum(),
        )
        print(" \n ")
        def count_duplicates(data, name):
            count = data.duplicated().sum()
            assert count == 0, f'{name} has {count} duplicates'

        # self.train_set['data'].reset_index(drop=True, inplace=True)
        count_duplicates(self.train_set['data'], 'input data')
        # First, split the data into two parts: 1/2 and 1/2
        train_set, temp_set = train_test_split(self.train_set, test_size=0.5, random_state=42)
        count_duplicates(train_set['data'], 'train data')
        count_duplicates(temp_set['data'], 'temp data')

        # Now split the temp_set into validation and holdout sets (statistical template set) with equal size
        temp_set['data'] = temp_set['data'].reset_index(drop=True)
        valid_set, holdout_set = train_test_split(temp_set, test_size=0.5, random_state=42)
        count_duplicates(valid_set['data'], 'valid data')
        count_duplicates(holdout_set['data'], 'holdout data')

        self.training_set = train_set
        self.valid_set = valid_set
        self.holdout_set = holdout_set

        del self.train_set

        def print_set_info(name, dataset):
            print(f"{name} Set:")
            print(f"{'-' * len(name)} ")
            print(f"  Data Shape:          {dataset['data'].shape}")
            print(f"  Labels Shape:        {dataset['labels'].shape}")
            print(f"  Weights Shape:       {dataset['weights'].shape}")
            print(f"  Sum Signal Weights:  {dataset['weights'][dataset['labels'] == 1].sum():.2f}")
            print(f"  Sum Background Weights: {dataset['weights'][dataset['labels'] == 0].sum():.2f}")
            print("\n")

        print_set_info("Training", self.training_set)
        print_set_info("Validation", self.valid_set)
        print_set_info("Holdout (For Statistical Template)", self.holdout_set)

        if XGBOOST:
            from boosted_decision_tree import BoostedDecisionTree

            self.model = BoostedDecisionTree()
            module_file = current_file + "/model_XGB.json"
            if os.path.exists(module_file):
                self.model.load(module_file)
                self.re_train = False  # if model is already trained, no need to retrain

            self.name = "model_XGB"

            print("Model is BDT")
        elif TENSORFLOW:
            from neural_network_TF import NeuralNetwork

            module_file = current_file + "/model_tf.keras"
            self.model = NeuralNetwork(train_data=self.training_set["data"])
            if os.path.exists(module_file):
                self.model.load(module_file)
                self.re_train = False  # if model is already trained, no need to retrain

            self.name = "model_tf"
            print("Model is TF NN")

        elif TORCH:
            from neural_network_torch import NeuralNetwork

            module_file = current_file + "/model_torch.pt"
            self.model = NeuralNetwork(train_data=self.train_set["data"])
            if os.path.exists(module_file):
                self.model.load(module_file)
                self.re_train = False  # if model is already trained, no need to retrain

            self.name = "model_torch"
            print("Model is Torch NN")

        if self.re_train:

            balanced_set = self.balance_set()

            if XGBOOST:
                self.model.fit(
                    balanced_set["data"], balanced_set["labels"], balanced_set["weights"],
                    valid_set=[self.valid_set["data"], self.valid_set["labels"], self.valid_set["weights"]],
                )
            else:
                self.model.fit(
                    balanced_set["data"], balanced_set["labels"], balanced_set["weights"]
                )
            self.model.save(current_file + "/" + self.name)
        
        self.stat_analysis = StatisticalAnalysis(self.model, self.holdout_set, template_file=os.path.join(current_file, TEMPLATE_FILE))
        saved_info_file = current_file + "/saved_info_" + self.name + ".pkl"
        if os.path.exists(saved_info_file) and not STAT_ONLY:
            self.stat_analysis.load(saved_info_file) 
        elif not STAT_ONLY:   
            self.stat_analysis.calculate_saved_info()
            self.stat_analysis.save(saved_info_file)

        def predict_and_analyze(dataset_name, data_set, fig_name, stat_only, syst_settings):
            score = self.model.predict(data_set["data"])
            results = self.stat_analysis.compute_mu(
                score,
                data_set["weights"],
                plot=fig_name,
                # stat_only=stat_only,
                # syst_fixed_setting=syst_settings,
            )

            print(f"{dataset_name} Results:")
            print(f"{'-' * len(dataset_name)} Results:")
            for key, value in results.items():
                print(f"\t{key} : {value}")
            print("\n")

        # Predict and analyze for each set
        datasets = [
            ("Training", self.training_set, "train_mu"),
            ("Validation", self.valid_set, "valid_mu"),
            ("Holdout", self.holdout_set, "holdout_mu"),
        ]

        for name, dataset, plot_name in datasets:
            predict_and_analyze(name, dataset, plot_name, stat_only=stat_only, syst_settings=syst_settings)
        
        # assume there were no problems with fitting; use valid and holdout sets to compute template histograms
        self.stat_analysis = StatisticalAnalysis(self.model, temp_set, template_file=os.path.join(current_file, TEMPLATE_FILE))
        self.stat_analysis.nominal_histograms(save=True)

        # delete the training set to free up memory
        del self.training_set
        del self.valid_set
        del self.holdout_set
        del self.stat_analysis.holdout_set

    def balance_set(self):
        balanced_set = self.training_set.copy()

        weights_train = self.training_set["weights"].copy()
        train_labels = self.training_set["labels"].copy()
        class_weights_train = (
            weights_train[train_labels == 0].sum(),
            weights_train[train_labels == 1].sum(),
        )

        for i in range(len(class_weights_train)):  # loop on B then S target
            # training dataset: equalize number of background and signal
            weights_train[train_labels == i] *= (
                    max(class_weights_train) / class_weights_train[i]
            )
            # test dataset : increase test weight to compensate for sampling

        balanced_set["weights"] = weights_train

        return balanced_set

    def predict(self, test_set, stat_only: bool = None, syst_settings: dict[str, float] = None):
        """
        Predicts the values for the test set.

        Params:
            stat_only (bool, optional): Force to compute stats only results [the highest priority]. Defaults to None.
            syst_settings (dict, optional): Dictionary containing the systematic settings of whether to fix systematics in fitting. For example, {'jes': True}. Defaults to None.

        Returns:
            dict: A dictionary with the following keys:
            - 'mu_hat': The predicted value of mu.
            - 'delta_mu_hat': The uncertainty in the predicted value of mu.
            - 'p16': The lower bound of the 16th percentile of mu.
            - 'p84': The upper bound of the 84th percentile of mu.
        """

        if NO_PREDICTION:
            return {
                'mu_hat': 1.,
                'delta_mu_hat': 0.,
                'p16': 1.,
                'p84': 1.,
            }
        test_data = test_set["data"]
        test_weights = test_set["weights"]

        predictions = self.model.predict(test_data)

        result = self.stat_analysis.compute_mu(
            predictions,
            test_weights,
            # stat_only=stat_only,
            # syst_fixed_setting=syst_settings
        )

        print("Test Results: ", result)

        # save the event counts with filename {number of events}_{time}.csv
        tmp_filename = os.path.join(self.counts_dir, f'{len(test_data)}_{int(time())}.csv')
        pd.DataFrame(result, index=[0]).assign(counts=len(test_data)).to_csv(
            os.path.join(self.counts_dir, f'{len(test_data)}.csv'),
            index=False,
            mode='a',
            header=not os.path.exists(os.path.join(self.counts_dir, f'{len(test_data)}.csv')),
        )

        return result


def train_test_split(data_set, test_size=0.2, random_state=42, reweight=True):
    data = data_set["data"].copy()
    train_set = {}
    test_set = {}
    full_size = len(data)

    print(f"Full size of the data is {full_size}")
    # combine everything into one dataframe
    for name in data_set.keys():
        if name != 'data':
            data[name] = data_set[name]
    # split
    train_df, test_df = sk_train_test_split(data, test_size=test_size, random_state=random_state)
    # rearrange the data into the original format
    train_set['detailed_labels'] = pd.Categorical(train_df.pop('detailed_labels'))
    test_set['detailed_labels'] = pd.Categorical(test_df.pop('detailed_labels'))
    for name in data_set.keys():
        if name not in ('data', 'detailed_labels'):
            train_set[name] = train_df.pop(name).to_numpy()
            test_set[name] = test_df.pop(name).to_numpy()
    train_set['data'] = train_df
    test_set['data'] = test_df

    print(f"Train size is {len(train_df)}")
    print(f"Test size is {len(test_df)}")

    if reweight is True:
        signal_weight = np.sum(data_set["weights"][data_set["labels"] == 1])
        background_weight = np.sum(data_set["weights"][data_set["labels"] == 0])
        signal_weight_train = np.sum(train_set["weights"][train_set["labels"] == 1])
        background_weight_train = np.sum(train_set["weights"][train_set["labels"] == 0])
        signal_weight_test = np.sum(test_set["weights"][test_set["labels"] == 1])
        background_weight_test = np.sum(test_set["weights"][test_set["labels"] == 0])

        train_set["weights"][train_set["labels"] == 1] = train_set["weights"][
                                                             train_set["labels"] == 1
                                                             ] * (signal_weight / signal_weight_train)
        test_set["weights"][test_set["labels"] == 1] = test_set["weights"][
                                                           test_set["labels"] == 1
                                                           ] * (signal_weight / signal_weight_test)

        train_set["weights"][train_set["labels"] == 0] = train_set["weights"][
                                                             train_set["labels"] == 0
                                                             ] * (background_weight / background_weight_train)
        test_set["weights"][test_set["labels"] == 0] = test_set["weights"][
                                                           test_set["labels"] == 0
                                                           ] * (background_weight / background_weight_test)

    return train_set, test_set
