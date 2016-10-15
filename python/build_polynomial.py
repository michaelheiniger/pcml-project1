# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.array(np.zeros((x.shape[0], degree+1)))
    for d in range(0,degree+1):
        phi[:, d] = np.power(x,d)
    return phi
