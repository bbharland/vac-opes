import numpy as np
import torch
import openmm.unit as unit
from scipy.stats import multivariate_normal   # Gaussian mixture model
from scipy.integrate import nquad             # computing bias shift


def silvermans_widths(weights):
    """Return the Gaussian kernel widths (bandwidth)
        * Parrinello: based on CVs estimated from unbiased simulation
        * Here: CVs have unit variance by construction (eigenfunctions)
        * multidimensional formula is Eq. 6 from Parrinello, https://dx.doi.org/10.1021/acs.jpclett.0c00497

    See: 2020-parrinello-opes.pdf, vac-opes.pdf
    """
    return silvermans_factor(weights) * np.ones(2)


def silvermans_factor(weights):
    """Scalar factor from Parrinello, Eq. 6 where d = num_cvs = 2.
    """
    n_eff = weights.sum() ** 2 / (weights ** 2).sum()
    return (1 / n_eff) ** (1 / 6)


def gaussian_height(widths):
    """Standard Gaussian height (h_0) for case where

        widths = [N_eff ** -(1/6)] diag(1, 1)

    https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    See: 2020-parrinello-opes.pdf, vac-opes.pdf
    """
    # assert len(widths) == 2, "widths must be d = 2"
    # assert widths[0] == widths[1], "sigma_1 must equal sigma_2"
    return 1 / (2 * np.pi * widths.prod())


class KDE:
    def __init__(self, centers, widths, heights):
        self.centers = centers
        self.widths = widths
        self.heights = heights

    def __len__(self):
        return len(self.heights)

    def __add__(self, other):
        return KDE(centers=np.vstack([self.centers, other.centers]),
                   widths=np.vstack([self.widths, other.widths]),
                   heights=np.concatenate([self.heights, other.heights]))

    def __call__(self, s):
        norm_sqs = np.sum(np.square((s - self.centers) / self.widths), axis=1)
        return np.sum(self.heights * np.exp(-0.5 * norm_sqs))

    def __eq__(self, other):
        return np.all(self.centers == other.centers) \
            and np.all(self.widths == other.widths) \
            and np.all(self.heights == other.heights)

    def renormalize(self):
        self.heights /= self.analytical_norm()
        return self

    def analytical_norm(self):
        """The analytical integral for the KDE distribution is

                I = (2 pi) sum_k^N h_k w_{1,k} w_{2,k}

        Use this to scale heights so that I = 1.0
        """
        total = 0
        for h, w in zip(self.heights, self.widths):
            total += h * w.prod()
        return 2 * np.pi * total


def kde_from_simdata(cvs, widths, weights):
    """Return KDE object using:

        cvs: np.ndarray with shape (num_frames, num_cvs)

        widths: np.ndarray with shape (num_cvs,)
            Current bandwidths given by Silverman's rule of thumb

        weights: np.ndarray with shape (num_frames,)
    """
    num_frames, num_cvs = cvs.shape
    assert widths.shape == (num_cvs,), \
    f'widths must be shape ({num_cvs},), found {widths.shape}'
    assert weights.shape == (num_frames,), \
    f'weights must be shape ({num_frames},), found {weights.shape}'

    return KDE(centers=cvs,
               widths=np.tile(widths, (num_frames, 1)),
               heights=gaussian_height(widths) * weights / weights.sum())


def compress(kde, dist_threshold=1):
    """Return a compressed KDE object.

    Reference: Supplemantary Information for Michele Invernizzi, Pablo M. Piaggi, and Michele Parrinello, "Unified Approach to Enhanced Sampling", Phys. Rev. X 10, 041034 (2020), https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.041034
    """
    gaussians = []

    for h, c, w in zip(kde.heights, kde.centers, kde.widths):
        gn = Gaussian(h, c, w)
        keep_merging = True

        while keep_merging:
            if len(gaussians) == 0:
                gaussians.append(gn)
                keep_merging = False
            else:
                dists = [g.distance(gn.c) for g in gaussians]
                idx = np.argmin(dists)
                if dists[idx] > dist_threshold:
                    gaussians.append(gn)
                    keep_merging = False
                else:
                    gn += gaussians[idx]
                    del gaussians[idx]

    return KDE(centers=np.vstack([g.c for g in gaussians]),
               widths=np.vstack([g.w for g in gaussians]),
               heights=np.stack([g.h for g in gaussians]))



class Gaussian:
    """Class for adding Gaussians and computing Mahalanobis distances between them.
    """
    def __init__(self, height, center, width):
        self.h = height # float
        self.c = center # ndarray with shape (num_cvs,)
        self.w = width  # ndarray with shape (num_cvs,)

    def __repr__(self):
        def arr_repr(a):
            return ', '.join([f'{val:.4f}' for val in a])

        height_str = f'height {self.h:.4f}'
        center_str = f'center [{arr_repr(self.c)}]'
        widths_str = f'widths [{arr_repr(self.w)}]'
        return f'Gaussian with {height_str}, {center_str}, {widths_str}'

    def __call__(self, s):
        return self.h * np.exp(-0.5 * self.distance(s)**2)

    def __add__(self, other):
        height = self.h + other.h
        center = (self.h * self.c + other.h * other.c) / height
        ws =  self.h * (self.w ** 2 + self.c ** 2)
        wo =  other.h * (other.w ** 2 + other.c ** 2)
        width = np.sqrt((ws + wo) / height - center ** 2)
        return Gaussian(height, center, width)

    def distance(self, s):
        return np.sqrt(np.sum(((s - self.c) / self.w) ** 2))


def bias_potential(p, kde):
    kT = unit.MOLAR_GAS_CONSTANT_R * p.temperature
    kT = kT.in_units_of(unit.kilojoule / unit.mole)._value
    scale = kT * (1 - 1 / p.bias_factor)
    regularization = p.dist_regularization
    return BiasPotential(kde, kT, scale, regularization, p.bias_factor)


class BiasPotential:
    def __init__(self, kde, kT, scale, regularization, bias_factor):
        self.kde = kde
        self.kT = kT
        self.scale = scale
        self.regularization = regularization
        self.bias_factor = bias_factor
        self.norm_factor = np.sum([kde(s) for s in kde.centers]) / len(kde)

    def __call__(self, s):
        return self.scale * np.log(self.kde(s) / self.norm_factor + self.regularization)

    def bias_exponential(self, s):
        return np.exp(-self(s) / self.kT)

    def bias_shift_quad(self, ranges, itype='i', return_err=False):
        """Return bias shift, c_n, computed by quadrature.

        Parameters
        ----------
        ranges : [[s1_min, s1_max], [s2_min, s2_max]]
        itype : str
            'w': use bias potential w(s) in integrand
            'i': use integrand written only in terms of KDE
        return_err : Bool
            Do you want the estimated error from scipy.integrate.nquad?
        """
        def integrand_w(s1, s2):
            s = np.array([s1, s2])
            return self.kde(s) * self.bias_exponential(s)

        def integrand_i(s1, s2):
            p = self.kde(np.array([s1, s2]))
            d = p / self.norm_factor + self.regularization
            e = 1 - 1 / self.bias_factor
            return p / d ** e

        if itype == 'w':
            integrand = integrand_w
        elif itype == 'i':
            integrand = integrand_i
        expectation, err = nquad(integrand, ranges)

        if return_err:
            return -self.kT * np.log(expectation), err
        else:
            return -self.kT * np.log(expectation)


def gaussian_mixture_model(kde):
    centers = kde.centers
    widths = kde.widths
    weights = np.array([2 * np.pi * w.prod() * h
                        for h, w in zip(kde.heights, widths)])
    return GaussianMixtureModel(centers, widths, weights)


class GaussianMixtureModel:
    """Switch from a KDE model to a GMM when you want to sample from the estimate of the unbiased distribution, p_0(s)

    GMM(s) = sum_k pi_k G(s, s_k) ,  where sum_k pi_k = 1

    Use weights as {pi_k}
    """
    def __init__(self, centers, widths, weights):
        self.choices = np.arange(len(self))
        self.weights = weights
        self.gaussians = [multivariate_normal(c, np.diag(w ** 2))
                          for c, w in zip(centers, widths)]

    def __call__(self, s):
        return np.sum([w * g.pdf(s)
                       for w, g in zip(self.weights, self.gaussians)])

    def __len__(self):
        return len(self.weights)

    def analytical_norm(self):
        return self.weights.sum()

    def random(self):
        """Draw single sample from GMM: s ~ p_GMM
        """
        index = self._select_gaussian()
        return self.gaussians[index].rvs()

    def random_batch(self, batch_size, shuffle=False):
        """Draw a batch of samples from GMM: s ~ p_GMM.  Procedure:
            1. select random Gaussian from GMM
            2. draw sample from it

        Parameters
        ----------
        batch_size : int
            Number of samples to draw at once
        shuffle : Bool
            Return the array of samples shuffled?
        """
        choices = self._select_gaussian(size=batch_size)
        choices, counts = np.unique(choices, return_counts=True)
        samples = np.vstack([self.gaussians[i].rvs(size=size)
                             for i, size in zip(choices, counts)])
        if shuffle:
            np.random.shuffle(samples)
        return samples

    def _select_gaussian(self, size=1):
        return np.random.choice(self.choices, size=size, p=self.weights)
