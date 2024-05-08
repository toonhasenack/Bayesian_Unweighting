import numpy as np
from scipy.special import xlogy
from typing import Tuple
from tqdm import tqdm 

class Unweight:
    def __init__(self, weights: np.ndarray, *chis: Tuple[int, np.ndarray]) -> None:
        """
        Initialize the Unweight class.

        Args:
            weights (np.ndarray): Array of weights.
            *chis (Tuple[int, np.ndarray]): Variable number of tuples containing power `n` and `chi` array.

        Raises:
            AssertionError: If lengths of `chi` arrays are not consistent.
        """
        self.chis = chis
        
        length = len(chis[0][1])
        for chi in chis:
            assert len(chi[1]) == length, "Not all chis have the same length!"

        if weights is None:
            self.weights = np.ones(length)
        else:
            self.weights = weights

    def entropy(self, p1, p2):
        """
        Calculate the entropy between two probability distributions.

        Args:
            p1 (np.ndarray): Probability distribution 1.
            p2 (np.ndarray): Probability distribution 2.

        Returns:
            float: Entropy value.
        """
        log = xlogy(p1, p1/p2)
        log[np.isnan(log)] = 0
        entropy = np.sum(log)
        return entropy
    
    def reweight(self, i: int = 0) -> None:
        """
        Perform reweighting.

        Args:
            i (int): Index of the chi array to reweight.
        """
        n, chi = self.chis[i]
        self.reweights = np.multiply(np.multiply(np.power(chi, n-1), np.exp(-1/2*np.power(chi,2.0))),self.weights)
        self.reweights = len(self.reweights)*self.reweights/np.sum(self.reweights)
        self.reprobs = self.reweights/np.sum(self.reweights)

    def unweight(self, Np: int) -> None:
        """
        Perform unweighting.

        Args:
            Np (int): Number of points.
        """
        pcum = np.zeros(len(self.reweights) + 1)
        pcum[1:] = np.cumsum(self.reprobs)
        unweights = np.zeros(len(self.reweights), dtype="int")
        for k in range(len(self.reweights)):
            for j in range(Np):
                condition_one = j/Np - pcum[k] >= 0
                condition_two = pcum[k+1] - j/Np >= 0
                if condition_one and condition_two:
                    unweights[k] += 1
        
        self.unweights = unweights
        self.unprobs = unweights/np.sum(unweights)

    def effective_replicas(self, weights):
        N = np.sum(weights)
        Neff = int(np.exp(-1/N*np.sum(xlogy(weights,weights/N))))
        return Neff

    def optimize(self, thresh: float, earlystopping: bool = True):
        """
        Optimize the unweighting process based on entropy threshold.

        Args:
            thresh (float): Entropy threshold value.

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: Tuple containing arrays of Nps, entropies, and optimal Np value.
        """
        Nps = np.logspace(1, np.log10(len(self.weights))+1, 50, dtype=np.int64)
        entropies = np.zeros(len(Nps))
        for i in tqdm(range(len(Nps))):
            self.unweight(Nps[i])
            entropies[i] = self.entropy(self.unprobs, self.reprobs)
            if entropies[i] <= thresh and earlystopping:
                loc = i
                break

        if i == len(Nps)-1:
            try:
                loc = np.where(entropies <= thresh)[0][0]
            except:
                print("Failed minimisation procedure! Defaulting to lowest entropy.")
                loc = -1
                
        Nopt = Nps[loc]

        return Nps, entropies, Nopt

    def iterate(self) -> None:
        """Perform iteration."""
        for i in range(len(self.chi)):
            self.reweight(i)
            self.unweight(10)
            self.weights = self.unweights
