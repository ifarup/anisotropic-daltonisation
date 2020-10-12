"""
daltonise: Algorithms for CVD simulation and daltonisation

Copyright (C) 2020 Ivar Farup

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.signal import correlate2d


def simulate(im, type='rg', degree=1):
    """
    Ultra-simple red-green and blue-yellow simulation

    The simulation is simply a linear transform performed in the current RGB
    colour space. For degree < 1, i.e., anomalous trichromats, a convex
    linear combination of the simulation and the identity is used.

    Paramters
    ---------
    type : str
        Either 'rg' (for red-green) or 'by' for blue-yellow
    degree : float
        Range 0-1, 0 is no CVD (identity matrix), 1 is dichromat
    """
    
    if type == 'rg':
        mat = np.array([[.5, .5, 0],
                        [.5, .5, 0],
                        [0, 0, 1]])
    elif type == 'by':
        mat = np.array([[.5, 0, .5],
                        [0, .5, .5],
                        [.25, .25, .5]])
    mat = degree * mat + (1 - degree) * np.eye(3)
    sim = np.einsum('ij,abj', mat, im)
    return sim


def unit_vectors(im, sim, el=[.3, .6, .1]):
    """
    Return principal vectors of the simulation.

    The first vector is taken as the lightness vector, the second is the
    first principal component of the difference between the image
    and the simulation, and the third is orthogonal to both of them.

    Paramters
    ---------
    im : ndarray
        The original image
    sim : ndarray
        The CVD simulation of the original image
    
    Returns
    -------
    el : ndarray
        The lightness vector
    ed : ndarray
        The difference vector
    ec : ndarray
        The chroma vector
    """
    
    el = el / np.linalg.norm(el)

    diff = np.reshape(im - sim, (im.shape[0] * im.shape[1], im.shape[2]))
    corr = np.dot(diff.T, diff)
    _, eig = np.linalg.eig(corr)
    ed = eig[0]
    ed = ed - np.dot(ed, el) * el
    ed = ed / np.linalg.norm(ed)

    ec = np.cross(ed, el)
    ec = ec / np.linalg.norm(ec)

    return el, ed, ec


def diff_filters(diff):
    """
    Compute different forward and backward FDM correlation filters.

    Parameters
    ----------
    diff : str
        finite difference method (FB, cent, Sobel, SobelFB, Feldman, FeldmanFB)

    Returns
    -------
    F_x : ndarray
    F_y : ndarray
    B_x : ndarray
    B_y : ndarray
    """

    if diff == 'FB':
        F_x = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        F_y = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
        B_x = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
        B_y = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
    elif diff == 'cent':
        F_x = .5 * np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        F_y = .5 * np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
        B_x = F_x
        B_y = F_y
    elif diff == 'Sobel':
        F_x = .125 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        F_y = .125 * np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        B_x = F_x
        B_y = F_y
    elif diff == 'SobelFB':
        F_x = .25 * np.array([[0, -1, 1], [0, -2, 2], [0, -1, 1]])
        F_y = .25 * np.array([[1, 2, 1], [-1, -2, -1], [0, 0, 0]])
        B_x = .25 * np.array([[-1, 1, 0], [-2, 2, 0], [-1, 1, 0]])
        B_y = .25 * np.array([[0, 0, 0], [1, 2, 1], [-1, -2, -1]])
    elif diff == 'Feldman':
        F_x = 1 / 32 * np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
        F_y = 1 / 32 * np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
        B_x = F_x
        B_y = F_y
    elif diff == 'FeldmanFB':
        F_x = 1 / 16 * np.array([[0, -3, 3], [0, -10, 10], [0, -3, 3]])
        F_y = 1 / 16 * np.array([[3, 10, 3], [-3, -10, -3], [0, 0, 0]])
        B_x = 1 / 16 * np.array([[-3, 3, 0], [-10, 10, 0], [-3, 3, 0]])
        B_y = 1 / 16 * np.array([[0, 0, 0], [3, 10, 3], [-3, -10, -3]])
    elif diff == 'circFB':
        x = (np.sqrt(2) - 1) / 2
        F_x = (np.array([[0, -x, x], [0, -1, 1], [0, -x, x]]) /
               (2 * x + 1))
        F_y = (np.array([[x, 1, x], [-x, -1, -x], [0, 0, 0]]) /
               (2 * x + 1))
        B_x = (np.array([[-x, x, 0], [-1, 1, 0], [-x, x, 0]]) /
               (2 * x + 1))
        B_y = (np.array([[0, 0, 0], [x, 1, x], [-x, -1, -x]]) /
               (2 * x + 1))
    return F_x, F_y, B_x, B_y


def diffusion_tensor(im, fx, fy, kappa, isotropic=False):
    """
    Compute the diffusion tensor for the given image.

    Parameters
    ----------
    im : ndarray
        The image
    fx : ndarray
        Convolution filter for the x component of the gradient
    fy : ndarray
        Convolution filter for the y component of the gradient
    kappa : float
        The diffusion parameter
    isotropic : bool
        Use isotropic instead of anisotropic diffusion

    Returns
    -------
    D11, D22, D12 : ndarray
        The components of the diffusion tensor
    """
    
    im = im.copy()

    gx = np.zeros(im.shape)
    gy = np.zeros(im.shape)

    if isotropic:

        for c in range(3):
            gx[..., c] = correlate2d(im.sum(2), fx, 'same', 'symm')
            gy[..., c] = correlate2d(im.sum(2), fy, 'same', 'symm')

        gradsq = (gx**2 + gy**2).sum(2)

        D11 = 1 / (1 + kappa * gradsq**2)
        D22 = D11.copy()
        D12 = np.zeros(D11.shape)
                    
    else:

        for c in range(3):

            gx[..., c] = correlate2d(im[..., c], fx, 'same', 'symm')
            gy[..., c] = correlate2d(im[..., c], fy, 'same', 'symm')

            S11 = (gx**2).sum(2)
            S12 = (gx * gy).sum(2)
            S22 = (gy**2).sum(2)

        # Eigenvalues and eigenvectors of the structure tensor

        lambda1 = .5 * (S11 + S22 + np.sqrt((S11 - S22)**2 + 4 * S12**2))
        lambda2 = .5 * (S11 + S22 - np.sqrt((S11 - S22)**2 + 4 * S12**2))

        theta1 = .5 * np.arctan2(2 * S12, S11 - S22)
        theta2 = theta1 + np.pi / 2

        v1x = np.cos(theta1)
        v1y = np.sin(theta1)
        v2x = np.cos(theta2)
        v2y = np.sin(theta2)

        # Diffusion tensor

        Dlambda1 = 1 / (1 + kappa * lambda1**2)
        Dlambda2 = 1 / (1 + kappa * lambda2**2)

        D11 = Dlambda1 * v1x**2 + Dlambda2 * v2x**2
        D22 = Dlambda1 * v1y**2 + Dlambda2 * v2y**2
        D12 = Dlambda1 * v1x * v1y + Dlambda2 * v2x * v2y

    return D11, D22, D12


def daltonise_simple(im, sim_function):
    """
    Simple baseline daltonisation algorithm. Mainly for use as initial value.

    Parameters
    ----------
    im : ndarray
        The input image
    sim_function : func
        The CVD simulation function

    Returns
    -------
    sdalt : ndarray
        The daltonised image
    """
    
    sim = sim_function(im)
    _, ed, ec = unit_vectors(im, sim)

    d = np.dot(im - sim, ed)

    dalt = im.copy()
    
    for i in range(3):
        dalt[..., i] += d * ec[i]
    
    dalt[dalt < 0] = 0
    dalt[dalt > 1] = 1

    return dalt


def construct_gradient(im, sim, fx, fy, simple=True):
    """
    Construct the gradient field for the daltonised image

    Parameters
    ----------
    im : ndarray
        The input image
    sim : ndarray
        The CVD simulation of the input image
    fx, fy : ndarray
        Convolution filters for the x and y components of the gradient
    simple : bool
        Use a simplified version of the gradient
    
    Returns
    -------
    Gx, Gy : ndarray
        The components of the constructed gradient field
    """
    
    _, ed, ec = unit_vectors(im, sim)

    gx = np.zeros(im.shape)
    gy = np.zeros(im.shape)

    for c in range(im.shape[2]):

        gx[..., c] = correlate2d(im[..., c], fx, 'same', 'symm')
        gy[..., c] = correlate2d(im[..., c], fy, 'same', 'symm')

    if simple: # as described in Farup, 2020

        Gx = gx + np.einsum('ijk,k,l', gx, ed, ec)
        Gy = gy + np.einsum('ijk,k,l', gy, ed, ec)

    else: # as described in Simon, J. Percept. Imag., 2018

        gsimx = np.zeros(im.shape)
        gsimy = np.zeros(im.shape)

        for c in range(im.shape[2]):

            gsimx[..., c] = correlate2d(sim[..., c], fx, 'same', 'symm')
            gsimy[..., c] = correlate2d(sim[..., c], fy, 'same', 'symm')

        ax = np.einsum('ijk,k,l', gx, ed, ec)
        ay = np.einsum('ijk,k,l', gy, ed, ec)
        a = np.einsum('ijk,ijk->ij', ax, ax) + np.einsum('ijk,ijk->ij', ay, ay)

        b = 2 * (np.einsum('ijk,ijk->ij', ax, gsimx) + np.einsum('ijk,ijk->ij', ay, gsimy))

        c = (np.einsum('ijk,ijk->ij', gsimx, gsimx) + np.einsum('ijk,ijk->ij', gsimy, gsimy) -
             np.einsum('ijk,ijk->ij', gx, gx) - np.einsum('ijk,ijk->ij', gy, gy))

        b2m4ac = b**2 - 4 * a * c
        b2m4ac[b2m4ac < 0] = 0
        a[a <= 0] == 1e-15

        chi_p = -b + np.sqrt(b2m4ac) / 2 * a
        chi_n = -b - np.sqrt(b2m4ac) / 2 * a
        print(chi_p.sum(), chi_n.sum())
        if abs(chi_p.sum()) < abs(chi_n.sum()):
            chi = chi_p
            print('chi_p')
        else:
            chi = chi_n
            print('chi_n')

        chi_stack = np.stack((chi, chi, chi), 2)

        Gx = gx + chi_stack * np.einsum('ijk,k,l', gx, ed, ec)
        Gy = gy + chi_stack * np.einsum('ijk,k,l', gy, ed, ec)
        
    return Gx, Gy


def daltonise_poisson(im, sim_function, nit=501, diff='FB', save=None, save_every=100):
    """
    Daltonise with the Poisson method of Simon and Farup, J. Percept. Imag., 2018

    Parameters
    ----------
    im : ndarray
        The input image to be daltonised
    sim_function : func
        The CVD simulation function
    nit : int
        Number of iterations
    diff : str
        The type of difference convolution filters (see diff_filters)
    save : str
        Filenamebase (e.g., 'im-%03d.png') or None
    save_evry : int
        Save every n iterations

    Returns
    -------
    pdalt : ndarray
        The daltonised image
    """
    
    sim = sim_function(im)
    sdalt = daltonise_simple(im, sim_function)
    fx, fy, bx, by = diff_filters(diff)
    Gx, Gy = construct_gradient(im, sim, fx, fy)
    gx = np.zeros(im.shape)
    gy = np.zeros(im.shape)
    pdalt = sdalt.copy()

    if save:
        plt.imsave(save % 0, pdalt)

    for i in range(nit):
        for c in range(im.shape[2]):
            gx[..., c] = correlate2d(pdalt[..., c], fx, 'same', 'symm')
            gy[..., c] = correlate2d(pdalt[..., c], fy, 'same', 'symm')
        
            pdalt[..., c] += .24 * (correlate2d(gx[..., c] - Gx[..., c], bx, 'same', 'symm') + 
                                    correlate2d(gy[..., c] - Gy[..., c], by, 'same', 'symm'))
        
        pdalt[pdalt < 0] = 0
        pdalt[pdalt > 1] = 1

        if save and i % save_every == 0:
            plt.imsave(save % i, pdalt)

    return pdalt


def daltonise_anisotropic(im, sim_function, nit=501, kappa=1e4, diff='FB',
                          save=None, save_every=100,
                          isotropic=False, debug=False):
    """
    Map the image to the spatial gamut defined by wp and bp.

    Parameters
    ----------
    im : ndarray (M x N x 3)
        The original image
    sim_function: func
        The CVD simulation function
    nit : int
        Number of iterations
    kappa : float
        anisotropy parameter
    diff : str
       finite difference method (FB, cent, Sobel, SobelFB, Feldman,
       FeldmanFB)
    isotropic : bool
       isotropic instead of anisotropi
    debug : bool
       print number of iterations every 10

    Returns
    -------
    adalt : ndarray
        The daltonised image
    """

    # Initialize

    sim = sim_function(im)
    adalt = daltonise_simple(im, sim_function)
    fx, fy, bx, by = diff_filters(diff)
    gx = np.zeros(im.shape)
    gy = np.zeros(im.shape)
    gxx = np.zeros(im.shape)
    gyy = np.zeros(im.shape)

    D11, D22, D12 = diffusion_tensor(im, fx, fy, kappa, isotropic)

    Gx, Gy = construct_gradient(im, sim, fx, fy)

    if save:
        plt.imsave(save % 0, adalt)

    # Iterate

    for i in range(nit):

        if (i % 10 == 0) and debug: print(i)

        # Anisotropic diffusion

        for c in range(3):
            gx[..., c] = correlate2d(adalt[..., c], fx, 'same', 'symm')
            gy[..., c] = correlate2d(adalt[..., c], fy, 'same', 'symm')
            gxx[..., c] = correlate2d(D11 * (gx[..., c] - Gx[..., c]) +
                                      D12 * (gy[..., c] - Gy[..., c]),
                                      bx, 'same', 'symm')
            gyy[..., c] = correlate2d(D12 * (gx[..., c] - Gx[..., c]) +
                                      D22 * (gy[..., c] - Gy[..., c]),
                                      by, 'same', 'symm')

        adalt += .24 * (gxx + gyy)

        adalt[adalt < 0] = 0
        adalt[adalt > 1] = 1

        # Save

        if save and i % save_every == 0:
            plt.imsave(save % i, adalt)

    return adalt
