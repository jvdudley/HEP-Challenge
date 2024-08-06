import numpy as np
from sys import path
from systematics import systematics
import pickle
from iminuit import Minuit
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    def __init__(self, model, holdout_set, bins=None, range=(0, 1), stat_only=None):
        self.model = model
        self.bins = bins
        self.range = range
        self.bin_edges = None
        self.holdout_set = holdout_set
        self.holdout_syst_applied = None
        # self.holdout_scores = None
        self.signal_hist = None
        self.background_hist = None
        self.signal_variance = None
        self.background_variance = None
        # stat_only argument is only there for compatibility with the old code
    
    def nominal_histograms(self, bins=None, apply_syst=False):
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
            # after some tests with the full test set, 128 seemed to be the best
            self.bins = 128
        self.bin_edges = np.linspace(*self.range, self.bins + 1)
        # determine if scores need to be computed
        if apply_syst != self.holdout_syst_applied:
            self.holdout_syst_applied = apply_syst
            # apply systematics
            holdout_syst = systematics(self.holdout_set) if apply_syst else self.holdout_set
            # compute scores
            holdout_scores = self.model.predict(holdout_syst['data'])
            self.signal_scores = holdout_scores[holdout_syst['labels'] == 1]
            self.background_scores = holdout_scores[holdout_syst['labels'] == 0]
            self.signal_weights = holdout_syst['weights'][holdout_syst['labels'] == 1]
            self.background_weights = holdout_syst['weights'][holdout_syst['labels'] == 0]

        # compute histograms
        self.signal_hist, _ = np.histogram(
            self.signal_scores,
            bins=self.bin_edges,
            weights=self.signal_weights,
        )
        self.background_hist, _ = np.histogram(
            self.background_scores,
            bins=self.bin_edges,
            weights=self.background_weights,
        )
        # compute variances
        self.signal_variance, _ = np.histogram(
            self.signal_scores,
            bins=self.bin_edges,
            weights=self.signal_weights**2,
        )
        self.background_variance, _ = np.histogram(
            self.background_scores,
            bins=self.bin_edges,
            weights=self.background_weights**2,
        )

        # consider returning whether or not there are too many bins

    def nominal_variance(self, scores, weights, plot=False):
        """
        Calculate the variance of the template from given scores and weights.
        """
        assert self.bin_edges is not None, "Must compute nominal histograms first."
        return np.histogram(scores, self.bin_edges, weights=weights**2)[0]


    def vary_hist_bins(self, bins_iter=None, data_set=None, apply_syst=False, mu_true=None, plot_title=None):
        """
        Compute histograms for each number of bins given in bins_iter.
        """
        if bins_iter is None:
            bins_iter = tqdm((2**j for j in range(11)), total=11)
        if data_set is None:
            data_set = self.holdout_set
            print('Using template set, ie fitting to itself.')
        if apply_syst:
            data_set = systematics(data_set)
        self.holdout_scores = None
        # compute scores
        scores = self.model.predict(data_set['data'])
        # loop over bins_iter
        results = []
        for bins in bins_iter:
            self.bins = bins
            self.nominal_histograms(apply_syst=apply_syst)
            results.append({
                'bins': bins,
                'prediction': self.compute_mu(scores, data_set['weights'])
            })
        # this should be changed to a weighted average using the variance
        mu_mean = np.mean([result['prediction']['mu_hat'] for result in results])
        mu_std = np.std([result['prediction']['mu_hat'] for result in results])
        mu_stderr = mu_std / np.sqrt(len(results))
        if plot_title is not None:
            plt.errorbar(
                [result['prediction']['mu_hat'] for result in results],
                [result['bins'] for result in results],
                xerr=[result['prediction']['delta_mu_hat'] / 2 for result in results],
                fmt='.',
            )
            if mu_true is not None:
                plt.axvline(mu_true, color='black', label=f'true $\mu$: {mu_true:.2f}')
            # plt.axvline(mu_mean, color='r', label=f'mean $\mu$: {mu_mean:.2f}')
            # plt.axvspan(mu_mean - mu_stderr, mu_mean + mu_stderr, color='r', alpha=.25, label=f'stderr: {mu_stderr:.2f}')
            plt.xlabel(r'$\mu$')
            plt.ylabel('bins')
            plt.yscale('log')
            plt.title(plot_title)
            plt.show()
        return results
    
    def estimate_mu(self, scores, weights=None, mu_range=None, mu_steps=None, epsilon=None, plot=False):
        """
        Estimate mu by scanning the likelihood.
        """
        if weights is None:
            weights = np.ones_like(scores)
        if epsilon is None:
            epsilon = np.spacing(np.zeros_like(weights[0]))
        def NLL(mu, observed, expected_signal, expected_background, epsilon=epsilon):
            """
            Negative log likelihood function.
            """
            expected = mu[:, None] * expected_signal + expected_background
            return np.sum(expected - observed * np.log(expected + epsilon), axis=1)
        
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
        NLL_values = NLL(mu_values, observed_hist, self.signal_hist, self.background_hist)
        # [
        #     NLL(mu, ...)
        #     for mu in mu_values
        # ]
        NLL_min = np.min(NLL_values)
        NLL_diff = NLL_values - NLL_min
        # compute mu_hat
        mu_hat = mu_values[np.argmin(NLL_diff)]
        p16 = mu_values[NLL_diff < .5][0]
        p84 = mu_values[NLL_diff < .5][-1]
        if plot:
            # plot histograms
            plt.stairs(
                self.signal_hist,
                self.bin_edges,
                fill=True,
                label='Signal',
            )
            plt.stairs(
                self.background_hist,
                self.bin_edges,
                baseline=self.signal_hist,
                fill=True,
                label='Background',
            )
            plt.stairs(
                observed_hist,
                self.bin_edges,
                label='Observed',
            )
            plt.legend()
            plt.title(plot + ': histograms')
            plt.show()
            # plot NLL
            plt.axhline(.5, color='r', linestyle='--')
            plt.axvspan(p16, p84, color='grey', alpha=.25, label=f'{p16:.2f} - {p84:.2f}')
            plt.axvline(mu_hat, color='C1', linestyle='--', label=f'mu_hat: {mu_hat:.2f}')
            plt.plot(mu_values, NLL_diff)
            plt.xlabel(r'$\mu$')
            plt.ylabel(r'$\Delta$NLL')
            plt.title(plot + ': NLL')
            plt.legend()
            plt.show()
        return (mu_hat, p16, p84)

    def compute_mu(self, scores, weights=None, mu_range=None, epsilon=None, plot=None):
        """
        Perform calculations to compute mu

        Args:
            scores (numpy.ndarray): Array of scores.
            weights (numpy.ndarray): Array of weights.
            plot (str, optional): Plot title. If None, do not plot. Defaults to None.
        
        Returns:
            dict: Dictionary containing calculated values of mu_hat, delta_mu_hat, p16, and p84.
        """
        if weights is None:
            weights = np.ones_like(scores)
        if mu_range is None:
            mu_range = (0, 4)
        mu_hat, p16, p84 = self.estimate_mu(scores, weights, mu_range=mu_range, mu_steps=10**4, epsilon=epsilon, plot=plot)
        delta_mu_hat = p84 - p16
        mu_range = (mu_hat - delta_mu_hat, mu_hat + delta_mu_hat)
        # mu_hat, p16, p84 = self.estimate_mu(scores, weights, mu_range=mu_range, mu_steps=10**4, epsilon=epsilon, plot=f'{plot}: mu final')
        return {
            'mu_hat': mu_hat,
            'delta_mu_hat': p84 - p16,
            'p16': p16,
            'p84': p84,
        }