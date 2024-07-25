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
    def __init__(self, model, holdout_set, bins=None, range=(0, 1)):
        self.model = model
        self.bins = bins
        self.range = range
        self.bin_edges = None
        self.holdout_set = holdout_set
        self.signal_hist = None
        self.background_hist = None
        self.signal_variance = None
        self.background_variance = None
    
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
        self.bin_edges = np.linspace(*self.range, self.bins + 1)
        # make postselection cuts
        holdout_syst = systematics(self.holdout_set.copy())
        # compute scores
        holdout_scores = self.model.predict(holdout_syst['data'])
        # compute histograms
        self.signal_hist, _ = np.histogram(
            holdout_scores[holdout_syst['labels'] == 1],
            bins=self.bin_edges,
            weights=holdout_syst['weights'][holdout_syst['labels'] == 1],
        )
        self.background_hist, _ = np.histogram(
            holdout_scores[holdout_syst['labels'] == 0],
            bins=self.bin_edges,
            weights=holdout_syst['weights'][holdout_syst['labels'] == 0],
        )
        # compute variances
        self.signal_variance, _ = np.histogram(
            holdout_scores[holdout_syst['labels'] == 1],
            self.bin_edges,
            weights=holdout_syst['weights'][holdout_syst['labels'] == 1]**2,
        )
        self.background_variance, _ = np.histogram(
            holdout_scores[holdout_syst['labels'] == 0],
            self.bin_edges,
            weights=holdout_syst['weights'][holdout_syst['labels'] == 0]**2,
        )

        # consider returning whether or not there are too many bins

    def nominal_variance(self, scores, weights, plot=False):
        """
        Calculate the variance of the template from given scores and weights.
        """
        assert self.bin_edges is not None, "Must compute nominal histograms first."
        return np.histogram(scores, self.bin_edges, weights=weights**2)[0]


    def determine_hist_bins(self, min_count=1):
        """
        Compute histograms and adjust the number of bins as needed.
        """
    
    def estimate_mu(self, scores, weights, mu_range=None, mu_steps=None, epsilon=None, plot=False):
        """
        Estimate mu by scanning the likelihood.
        """
        def NLL(mu, observed, expected_signal, expected_background, epsilon=epsilon):
            """
            Negative log likelihood function.
            """
            expected = mu * expected_signal + expected_background
            if epsilon is None:
                epsilon = np.spacing(np.zeros_like(expected[0]))
            return np.sum(expected - observed * np.log(expected + epsilon))
        
        if mu_range is None:
            mu_range = (0, 3)
        if mu_steps is None:
            mu_steps = 10**4 # decrease this later and use multiple steps to decrease range
        mu_values = np.linspace(*mu_range, mu_steps)
        # compute template histograms
        if self.signal_hist is None or self.background_hist is None:
            self.nominal_histograms()
        # compute observed histogram
        observed_hist, _ = np.histogram(scores, bins=self.bin_edges, weights=weights)
        # compute negative log likelihoods
        NLL_values = [
            NLL(mu, observed_hist, self.signal_hist, self.background_hist)
            for mu in mu_values
        ]
        NLL_min = np.min(NLL_values)
        NLL_diff = NLL_values - NLL_min
        # compute mu_hat
        mu_hat = mu_values[np.argmin(NLL_diff)]
        p16 = mu_values[NLL_diff < .5][0]
        p84 = mu_values[NLL_diff < .5][-1]
        if plot:
            plt.axhline(.5, color='r', linestyle='--')
            plt.axvspan(p16, p84, color='grey', alpha=.25, label=f'{p16:.2f} - {p84:.2f}')
            plt.axvline(mu_hat, color='C1', linestyle='--', label=f'mu_hat: {mu_hat:.2f}')
            plt.plot(mu_values, NLL_diff)
            plt.xlabel(r'$\mu$')
            plt.ylabel(r'$\Delta$NLL')
            plt.legend()
            plt.show()
        return (mu_hat, p16, p84)

    def compute_mu(self, scores, weights, mu_range=None, epsilon=None, plot=False):
        """
        Perform calculations to compute mu

        Args:
            scores (numpy.ndarray): Array of scores.
            weights (numpy.ndarray): Array of weights.
            plot (bool, optional): Whether to plot the profile likelihood. Defaults to False.
        
        Returns:
            dict: Dictionary containing calculated values of mu_hat, delta_mu_hat, p16, and p84.
        """
        if mu_range is None:
            mu_range = (0, 4)
        mu_hat, p16, p84 = self.estimate_mu(scores, weights, mu_range=mu_range, mu_steps=1000, epsilon=epsilon, plot=plot)
        delta_mu_hat = p84 - p16
        mu_range = (mu_hat - delta_mu_hat, mu_hat + delta_mu_hat)
        if plot:
            plt.hist(
                [self.bin_edges[:-1],] * 2, # one each for sig and bkg
                self.bin_edges,
                weights=[self.background_hist, self.signal_hist],
                histtype='barstacked',
                label=['Background', 'Signal'],
            )
            plt.legend()
            plt.show()
        mu_hat, p16, p84 = self.estimate_mu(scores, weights, mu_range=mu_range, mu_steps=10**4, epsilon=epsilon, plot=plot)
        return {
            'mu_hat': mu_hat,
            'delta_mu_hat': p84 - p16,
            'p16': p16,
            'p84': p84,
        }