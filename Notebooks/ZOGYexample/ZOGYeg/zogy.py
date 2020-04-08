#!/usr/bin/env python3

############################################################
# Python implementation of ZOGY image subtraction algorithm
# See Zackay, Ofek, and Gal-Yam 2016 for details
# http://arxiv.org/abs/1601.02655
# SBC - 6 July 2016
# FJM - 20 October 2016
# SBC - 28 July 2017
# FJM - 12 June 2018
# SR  - 08 April 2020
############################################################

import sys, time
import numpy as np

import astropy.io.fits as fits

# Could also use numpy.fft, but this is apparently faster
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1.)

def py_zogy(Nf, Rf, P_Nf, P_Rf, S_Nf, S_Rf, SN, SR, dx=0.25, dy=0.25):

    '''Python implementation of ZOGY image subtraction algorithm.
    As per Frank's instructions, will assume images have been aligned,
    background subtracted, and gain-matched.
    
    Arguments:
    N: New image (filename)
    R: Reference image (filename)
    P_N: PSF of New image (filename)
    P_R: PSF or Reference image (filename)
    S_N: 2D Uncertainty (sigma) of New image (filename)
    S_R: 2D Uncertainty (sigma) of Reference image (filename)
    SN: Average uncertainty (sigma) of New image
    SR: Average uncertainty (sigma) of Reference image
    dx: Astrometric uncertainty (sigma) in x coordinate
    dy: Astrometric uncertainty (sigma) in y coordinate
    
    Returns:
    D: Subtracted image
    P_D: PSF of subtracted image
    S_corr: Corrected subtracted image
        D_nocorr: Subtracted image with no correction for correlated noise
        P_Dnocorr: PSF corresponding to subtracted image D_nocorr
    '''
    
    # Load the new and ref images into memory
    N = fits.open(Nf)[0].data
    R = fits.open(Rf)[0].data
    
    # Load the PSFs into memory
    P_N_small = fits.open(P_Nf)[0].data
    P_R_small = fits.open(P_Rf)[0].data
    
    # Place PSF at center of image with same size as new / reference
    P_N = np.zeros(N.shape)
    P_R = np.zeros(R.shape)

    idx = [slice(int(N.shape[0]/2) - int(P_N_small.shape[0]/2),
                 int(N.shape[0]/2) + int(P_N_small.shape[0]/2) + 1),
           slice(int(N.shape[1]/2) - int(P_N_small.shape[1]/2),
                 int(N.shape[1]/2) + int(P_N_small.shape[1]/2) + 1)]

    P_N[tuple(idx)] = P_N_small
    P_R[tuple(idx)] = P_R_small
        
    # Shift the PSF to the origin so it will not introduce a shift
    P_N = fft.fftshift(P_N)
    P_R = fft.fftshift(P_R)
        
    # Take all the Fourier Transforms
    N_hat = fft.fft2(N)
    R_hat = fft.fft2(R)
    
    P_N_hat = fft.fft2(P_N)
    P_R_hat = fft.fft2(P_R)
    
    # Fourier Transform of Difference Image (Equation 13)
    D_hat_num = (P_R_hat * N_hat - P_N_hat * R_hat) 
    D_hat_den = np.sqrt(SN**2 * np.abs(P_R_hat**2) + SR**2 * np.abs(P_N_hat**2))
    D_hat = D_hat_num / D_hat_den
    
    # Flux-based zero point (Equation 15)
    FD = 1. / np.sqrt(SN**2 + SR**2)
    
    # Difference image corrected for correlated noise
    D = np.real(fft.ifft2(D_hat)) / FD

    # Difference image not corrected for correlated noise
    D_nocorr = np.real(fft.ifft2(D_hat_num))
    
    # Fourier Transform of PSF of Subtraction Image (Equation 14)
    P_D_hat = P_R_hat * P_N_hat / FD / D_hat_den
    
    # PSF of Subtraction Image D
    P_D = np.real(fft.ifft2(P_D_hat))
    P_D = fft.ifftshift(P_D)    
    P_D = P_D[tuple(idx)]

    # PSF of Subtraction Image D_nocorr
    P_Dnocorr = np.real(fft.ifft2(P_R_hat * P_N_hat))
    P_Dnocorr = fft.ifftshift(P_Dnocorr)
    P_Dnocorr = P_Dnocorr[tuple(idx)]
    
    # Fourier Transform of Score Image (Equation 17)
    S_hat = FD * D_hat * np.conj(P_D_hat)
    
    # Score Image
    S = np.real(fft.ifft2(S_hat))
    
    # Now start calculating Scorr matrix (including all noise terms)
    
    # Start out with source noise
    # Load the sigma images into memory
    S_N = fits.open(S_Nf)[0].data
    S_R = fits.open(S_Rf)[0].data

    # Sigma to variance
    V_N = S_N**2
    V_R = S_R**2
    
    # Fourier Transform of variance images
    V_N_hat = fft.fft2(V_N)
    V_R_hat = fft.fft2(V_R)
    
    # Equation 28
    kr_hat = np.conj(P_R_hat) * np.abs(P_N_hat**2) / (D_hat_den**2)
    kr = np.real(fft.ifft2(kr_hat))
    
    # Equation 29
    kn_hat = np.conj(P_N_hat) * np.abs(P_R_hat**2) / (D_hat_den**2)
    kn = np.real(fft.ifft2(kn_hat))
    
    # Noise in New Image: Equation 26
    V_S_N = np.real(fft.ifft2(V_N_hat * fft.fft2(kn**2)))
    
    # Noise in Reference Image: Equation 27
    V_S_R = np.real(fft.ifft2(V_R_hat * fft.fft2(kr**2)))
    
    # Astrometric Noise
    # Equation 31
    # TODO: Check axis (0/1) vs x/y coordinates
    S_N = np.real(fft.ifft2(kn_hat * N_hat))
    dSNdx = S_N - np.roll(S_N, 1, axis=1)
    dSNdy = S_N - np.roll(S_N, 1, axis=0)
   
    # Equation 30
    V_ast_S_N = dx**2 * dSNdx**2 + dy**2 * dSNdy**2

    # Equation 33
    S_R = np.real(fft.ifft2(kr_hat * R_hat))
    dSRdx = S_R - np.roll(S_R, 1, axis=1)
    dSRdy = S_R - np.roll(S_R, 1, axis=0)
  
    # Equation 32
    V_ast_S_R = dx**2 * dSRdx**2 + dy**2 * dSRdy**2
      
    # Calculate Scorr
    S_corr = S / np.sqrt(V_S_N + V_S_R + V_ast_S_N + V_ast_S_R)
      
    return D, P_D, S_corr, D_nocorr, P_Dnocorr
    
if __name__ == "__main__":
    t0 = time.time()
    if len(sys.argv) == 16:
    
        D, P_D, S_corr, D_nocorr, P_Dnocorr = \
                         py_zogy(sys.argv[1], sys.argv[2], sys.argv[3],
                         sys.argv[4], sys.argv[5], sys.argv[6],
                         float(sys.argv[7]), float(sys.argv[8]), 
                         dx=float(sys.argv[9]), dy=float(sys.argv[10]))
        
        # Difference image D
        tmp = fits.open(sys.argv[1])
        tmp[0].data = D.astype(np.float32)
        tmp.writeto(sys.argv[11], output_verify="warn", overwrite=True)

        # Difference image D_nocorr
        tmp[0].data = D_nocorr.astype(np.float32)
        tmp.writeto(sys.argv[14], output_verify="warn", overwrite=True)

        # S_corr image
        tmp[0].data = S_corr.astype(np.float32)
        tmp.writeto(sys.argv[13], output_verify="warn", overwrite=True)
        
        # PSF image for D
        tmp = fits.open(sys.argv[3])
        tmp[0].data = P_D.astype(np.float32)
        tmp.writeto(sys.argv[12], output_verify="warn", overwrite=True)

        # PSF Image for D_nocorr
        tmp[0].data = P_Dnocorr.astype(np.float32)
        tmp.writeto(sys.argv[15], output_verify="warn", overwrite=True)
        t1 = time.time()
        print(f"time elapsed: {t1-t0:.2f} seconds")
    else:
        print("Usage: python3 zogy.py <NewImage> <RefImage> <NewPSF> <RefPSF> <NewSigmaImage> <RefSigmaImage> <NewSigmaMode> <RefSigmaMode> <AstUncertX> <AstUncertY> <DiffImage> <DiffPSF> <ScorrImage> <DiffImageNoCorr> <DiffPSFNoCorr>")
