"""
Store functions related to the reconstruction of parameters
"""

import numpy as np

from tqdm import tqdm


def compute_error(bundle):
    """
    Compute relative error of reconstruction.

    Arguments
        bundle
            Dict of original and reconstructed data
    """
    rel_err_A = abs(bundle.orig_A - bundle.rec_A) / (1 + abs(bundle.orig_A))
    rel_err_B = abs(bundle.orig_B - bundle.rec_B) / (1 + abs(bundle.orig_B))

    return rel_err_A, rel_err_B

def find_tpi_crossings(series):
    """
    Find indices where time series wraps from 2pi back to 0.

    Arguments:
        series
            List of scalar values
    """
    idx = []
    for i in range(len(series)-1):
        if series[i] > 3/2*np.pi and series[i+1] < 1/2*np.pi:
            idx.append(i)
    return idx

class Reconstructor(object):
    """
    Reconstruct system parameters from evolution observations
    """
    def __init__(self, ts, data):
        """
        Initialize reconstruction process.

        Arguments
            ts
                List of time points of the simulation
            data
                List of (OMEGA, [phase differences]) pairs
        """
        self._ts = ts
        self._data = data

        pdiffs = data[0][1]
        self._graph_shape = (len(pdiffs), len(pdiffs))

    @property
    def ts(self):
        return self._ts

    @property
    def data(self):
        return self._data

    @staticmethod
    def extract_phase_differences(sols, ts, driver):
        """
        Extract phase difference of each oscillator to driving force in locked state.

        Arguments:
            sols
                List of system solutions
            ts
                List of time points of the simulation
            driver
                Function of driver of system
        """
        sols %= 2*np.pi

        # find driver's phase
        driver_theta = driver(ts) % (2*np.pi)
        driver_idx = find_tpi_crossings(driver_theta)

        # compute phase differences
        driver_ref = driver_theta[-1]

        pdiffs = []
        for sol in sols:
            d = driver_ref - sol[-1]

            # wrap d if necessary
            if d < 0:
                d = 2*np.pi + d

            pdiffs.append(d)

        return pdiffs

    def _mat_to_flat(self, mat):
        """
        Convert matrix to flat version without diagonal entries.

        Arguments:
            mat
                Matrix to be flattened
        """
        # find indices of entries which are not on the diagonal
        gsize = self._graph_shape[0]**2
        nondiag_inds = np.where(
            np.arange(gsize) % (self._graph_shape[0]+1) != 0)

        return mat.reshape(gsize)[nondiag_inds]

    def _flat_to_mat(self, flat, repl=0):
        """
        Convert flat version to matrix and reinsert diagonal elements.

        Arguments:
            flat
                Flat list to be converted
            repl
                Scalar to be inserted into diagonal
        """
        # find indices of entries which are on the diagonal
        gsize = self._graph_shape[0]**2
        diag_inds = np.where(
            np.arange(gsize) % (self._graph_shape[0]+1) == 0)

        # insert
        for i in diag_inds[0]:
            flat = np.insert(flat, i, repl)

        return flat.reshape(self._graph_shape)

    def _compute_A(self, i, phase_diffs):
        """
        Compute flat representation of coupling terms
        """
        coeffs_A = np.zeros(self._graph_shape)

        tmp = np.reshape(phase_diffs, (len(phase_diffs), 1))
        coeffs_A[i,:] = np.sin(tmp - tmp.transpose())[i,:]

        return self._mat_to_flat(coeffs_A)

    def _compute_B(self, i, phase_diffs):
        """
        Compute terms of external force coupling
        """
        coeffs_B = np.zeros(self._graph_shape[0])
        coeffs_B[i] = np.sin(phase_diffs[i])

        return coeffs_B

    def reconstruct(self):
        """
        Reconstruct parameters from provided data
        """
        # assemble linear system
        rhs = []
        lhs = []
        for (OMEGA, omega), phase_diffs in tqdm(self.data):
            for i in range(len(phase_diffs)):
                # coefficient matrix
                flat_A = self._compute_A(i, phase_diffs)
                flat_B = self._compute_B(i, phase_diffs)

                row = flat_A.tolist() + flat_B.tolist()
                rhs.append(row)

                # solution vector
                theta_dot = OMEGA - omega
                lhs.append(theta_dot)

        print('Using {} data points to solve system of {} variables (rank: {})'.format(
            len(rhs),
            self._graph_shape[0]**2,
            np.linalg.matrix_rank(rhs)))

        # solve system
        x = np.linalg.lstsq(rhs, lhs)[0]

        # extract reconstructed parameters from solution
        rec_A = self._flat_to_mat(x[:-self._graph_shape[0]])
        rec_B = x[-self._graph_shape[0]:]

        return rec_A, rec_B
