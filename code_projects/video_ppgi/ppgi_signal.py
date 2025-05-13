import numpy as np


def extract_bvp_POS(rgb_full: np.ndarray, fs: float, **kwargs) -> np.ndarray:
    """Implementation of plane-orthogonal-to-skin based blood volume pulse extraction introduced by Wang et al. in:
    W. Wang, A. C. Den Brinker, S. Stuijk, and G. de Haan, “Algorithmic Principles of Remote PPG,” IEEE transactions on bio-medical engineering, vol. 64, no. 7, pp. 1479–1491, 2017, doi: 10.1109/TBME.2016.2609282.
    Returns
    -------
    numpy.ndarray of shape [n, 1] where n matches the size of the original rgb_seq
        the blood volume pulse extracted from the rgb_seq
    """
    _tmp_norm_len = 2  # Interval size in seconds to use for the temporal normalization. The value is chosen according to Wang et al. in: W. Wang, A. C. Den Brinker, S. Stuijk, and G. de Haan, “Algorithmic Principles of Remote PPG,” IEEE transactions on bio-medical engineering, vol. 64, no. 7, pp. 1479–1491, 2017, doi: 10.1109/TBME.2016.2609282.

    tmp_norm_n_frames = int(np.ceil(fs * _tmp_norm_len))
    # projection matrix used by POS
    pm = np.array([
        [0, 1, -1],
        [-2, 1, 1]
    ])
    # compute bvp
    n = rgb_full.shape[0]  # number of sample points, i.e. frames
    h = np.zeros((1, n))
    c = rgb_full.copy()
    # loop over overlapping windows
    for i in range(n):
        m = i - tmp_norm_n_frames
        if m >= 0:
            # temporal normalization
            cn = c[m:i] / np.mean(c[m:i], axis=0)
            # projection
            s = np.matmul(pm, np.transpose(cn))
            s1 = s[0, :]
            s2 = s[1, :]
            if s2.std() == 0:
                print("! s2.std() is zero !")
                return None
            # tuning
            hi = s1 + (s1.std() / s2.std()) * s2
            # overlap-adding
            h[0, m:i] = h[0, m:i] + (hi - hi.mean())

    return h.squeeze()[:, np.newaxis]