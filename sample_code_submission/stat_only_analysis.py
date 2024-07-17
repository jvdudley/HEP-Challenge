import numpy as np
from sys import path
from systematics import systematics
import pickle
from iminuit import Minuit
import matplotlib.pyplot as plt


path.append("../")
path.append("../ingestion_program")


class StatOnlyAnalysis:
    """
    A class that performs statistical analysis using a given model and holdout set.

    Args:
        model: The model used for prediction.
        holdout_set (dict): Dictionary containing holdout set data and labels.
        bins (int, optional): Number of bins for histogram calculation. Defaults to 10.
    
    Attributes:
        model: The model used for prediction.
        bins (int): Number of bins for histogram calculation.
        bin_edges (numpy.ndarray): Array of bin edges.
        holdout_set (dict): Dictionary containing holdout set data and labels.
    
    Methods:
        compute_mu: Perform calculations to calculate mu.
        nominal_histograms: Calculate the nominal histograms for signal and background events.
    """
    def __init__(self, model, holdout_set, bins=None):
        self.model = model
        self.bins = bins
        self.holdout_set = holdout_set
        self.signal_hist = None
        self.background_hist = None
    
    def nominal_histograms(self, bins=None):
        """
        Calculate the nominal histograms for signal and background events.

        Parameters:
        - alpha (float): The value of the systematic parameter.

        Returns:
        - holdout_signal_hist (numpy.ndarray): The histogram of signal events in the holdout set.
        - holdout_background_hist (numpy.ndarray): The histogram of background events in the holdout set.
        """
        # if bins is None, should compute optimal number of bins
        if self.bins is None:
            # TMP: use 2000 bins for now
            self.bins = 2000
        self.bin_edges = np.linspace(0, 1, self.bins + 1)
        # make postselection cuts
        holdout_syst = systematics(self.holdout_set.copy())
        # compute scores
        holdout_scores = self.model.predict(holdout_syst['data'])
        # compute histograms
        holdout_signal_hist, _ = np.histogram(
            holdout_scores[holdout_syst['labels'] == 1],
            bins=self.bin_edges,
            weights=holdout_syst['weights'][holdout_syst['labels'] == 1],
        )
        holdout_background_hist, _ = np.histogram(
            holdout_scores[holdout_syst['labels'] == 0],
            bins=self.bin_edges,
            weights=holdout_syst['weights'][holdout_syst['labels'] == 0],
        )
        self.signal_hist = holdout_signal_hist
        self.background_hist = holdout_background_hist
        # consider returning whether or not there are too many bins

    def determine_hist_bins(self):
        """
        Compute histograms and adjust the number of bins as needed.
        """
    
    def compute_mu(self, scores, weights, plot=False):
        """
        Perform calculations to compute mu

        Args:
            scores (numpy.ndarray): Array of scores.
            weights (numpy.ndarray): Array of weights.
            plot (bool, optional): Whether to plot the profile likelihood. Defaults to False.
        
        Returns:
            dict: Dictionary containing calculated values of mu_hat, delta_mu_hat, p16, and p84.
        """