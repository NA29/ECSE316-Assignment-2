#!/usr/bin/env python3

import argparse
import time
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# If you prefer cv2, you can swap the image loading code to use it.
# import cv2

# ============================
# CONFIG
# ============================

# Change this to the filename of the image your prof gave you
DEFAULT_IMAGE = "moonlanding.png"

# For runtime plots (mode 4)
MIN_POWER = 5          # 2^5 = 32
MAX_POWER = 7          # 2^7 = 128; increase to 10 if your computer is fast
NUM_TRIALS = 10        # number of repetitions per size


# ============================
# 1D DFT (naïve, O(N^2))
# ============================

def dft_1d(x):
    """
    Naïve 1D Discrete Fourier Transform.
    x: 1D array-like of real or complex numbers.
    Returns: 1D numpy array of complex numbers.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    X = np.zeros(N, dtype=np.complex128)

    # Direct implementation of the definition
    for k in range(N):
        n = np.arange(N)
        exponent = -2j * np.pi * k * n / N
        X[k] = np.sum(x * np.exp(exponent))

    return X


# ============================
# 1D FFT (Cooley–Tukey, O(N log N))
# ============================

def fft_1d(x):
    """
    Recursive Cooley–Tukey FFT.
    Assumes len(x) is a power of 2.
    x: 1D array-like of real or complex numbers.
    Returns: 1D numpy array of complex numbers.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]

    if N == 1:
        return x.copy()

    if N % 2 != 0:
        raise ValueError(f"fft_1d length must be a power of 2 (got N = {N}).")

    # Split into even and odd indices
    X_even = fft_1d(x[0::2])
    X_odd = fft_1d(x[1::2])

    # Twiddle factors: W_N^k = e^{-j 2πk / N}
    k = np.arange(N // 2)
    twiddle = np.exp(-2j * np.pi * k / N) * X_odd

    # Combine
    X = np.zeros(N, dtype=np.complex128)
    X[:N // 2] = X_even + twiddle
    X[N // 2:] = X_even - twiddle

    return X


# ============================
# 1D Inverse FFT
# ============================

def ifft_1d(X):
    """
    Inverse FFT using conjugate trick.
    Assumes len(X) is a power of 2.
    X: 1D array-like of complex numbers.
    Returns: 1D numpy array of complex numbers (time/spatial domain).
    """
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]

    X_conj = np.conjugate(X)
    y = fft_1d(X_conj)
    x = np.conjugate(y) / N

    return x


# ============================
# Helpers: power-of-two padding
# ============================

def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n (for n >= 1)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 << (n - 1).bit_length()


def pad_to_power_of_two_2d(arr):
    """
    Pad a 2D array with zeros so that both dimensions are powers of 2.
    arr: 2D array-like.
    Returns: (padded_array (complex128), original_shape).
    """
    arr = np.asarray(arr)
    rows, cols = arr.shape

    new_rows = _next_power_of_two(rows)
    new_cols = _next_power_of_two(cols)

    if new_rows == rows and new_cols == cols:
        # Already powers of 2
        return arr.astype(np.complex128), (rows, cols)

    padded = np.zeros((new_rows, new_cols), dtype=np.complex128)
    padded[:rows, :cols] = arr

    return padded, (rows, cols)


def crop_to_original_2d(arr, original_shape):
    """
    Crop a padded 2D array back to original shape.
    arr: 2D numpy array.
    original_shape: (rows, cols) tuple from pad_to_power_of_two_2d.
    """
    rows, cols = original_shape
    return arr[:rows, :cols]


# ============================
# 2D FFT / IFFT
# ============================

def fft_2d(mat):
    """
    2D FFT by applying 1D FFT on rows then on columns.
    Assumes both dimensions are powers of 2.
    mat: 2D array-like (real or complex).
    Returns: 2D numpy array of complex numbers (frequency domain).
    """
    A = np.asarray(mat, dtype=np.complex128)
    rows, cols = A.shape

    # FFT along rows
    temp = np.zeros_like(A, dtype=np.complex128)
    for i in range(rows):
        temp[i, :] = fft_1d(A[i, :])

    # FFT along columns
    F = np.zeros_like(temp, dtype=np.complex128)
    for j in range(cols):
        F[:, j] = fft_1d(temp[:, j])

    return F


def ifft_2d(F):
    """
    2D Inverse FFT by applying 1D IFFT on columns then on rows.
    Assumes both dimensions are powers of 2.
    F: 2D array-like of complex numbers.
    Returns: 2D numpy array of complex numbers (spatial domain).
    """
    F = np.asarray(F, dtype=np.complex128)
    rows, cols = F.shape

    # IFFT along columns
    temp = np.zeros_like(F, dtype=np.complex128)
    for j in range(cols):
        temp[:, j] = ifft_1d(F[:, j])

    # IFFT along rows
    A = np.zeros_like(temp, dtype=np.complex128)
    for i in range(rows):
        A[i, :] = ifft_1d(temp[i, :])

    return A


# ============================
# Naïve 2D DFT (for runtime comparison)
# ============================

def dft_2d(mat):
    """
    2D DFT using separability:
    apply 1D DFT on rows, then on columns.
    This is O(N^3) with our vectorized 1D DFT.
    """
    A = np.asarray(mat, dtype=np.complex128)
    rows, cols = A.shape

    # DFT along rows
    temp = np.zeros_like(A, dtype=np.complex128)
    for i in range(rows):
        temp[i, :] = dft_1d(A[i, :])

    # DFT along columns
    F = np.zeros_like(temp, dtype=np.complex128)
    for j in range(cols):
        F[:, j] = dft_1d(temp[:, j])

    return F


# ============================
# Image utilities
# ============================

def load_image_gray(path):
    """
    Load image and convert to grayscale float in [0,1].
    Uses matplotlib.image.imread to avoid extra dependencies.
    """
    img = plt.imread(path)

    # If PNG, it may come in as uint8 or float in [0,1]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    # Convert to grayscale if RGB or RGBA
    if img.ndim == 3:
        # If RGBA, ignore alpha
        img = img[..., :3]
        img = img.mean(axis=2)

    return img


def to_displayable_image(arr):
    """
    Convert complex/float array to displayable grayscale [0,1].
    """
    arr = np.real(arr)
    # Simple normalization: shift & scale to [0,1]
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < 1e-12:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def fft_magnitude_log(F):
    """
    Compute log-scaled magnitude of FFT coefficients for plotting.
    """
    mag = np.abs(F)
    # Avoid log(0)
    return np.log1p(mag)


def fft_shift_2d(F):
    """
    Center the zero-frequency component (like np.fft.fftshift).
    """
    return np.fft.fftshift(F)


# ============================
# MODE 1: FFT visualization
# ============================

def run_mode_1(image_path: str):
    """
    Mode 1:
    - Load image
    - Pad to powers of two
    - Compute 2D FFT
    - Show original + log-scaled FFT magnitude
    """
    img = load_image_gray(image_path)
    padded, orig_shape = pad_to_power_of_two_2d(img)

    F = fft_2d(padded)
    F_shift = fft_shift_2d(F)
    log_mag = fft_magnitude_log(F_shift)

    # Crop FFT magnitude back to original shape for display
    log_mag_cropped = crop_to_original_2d(log_mag, orig_shape)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Log-magnitude FFT")
    plt.imshow(log_mag_cropped, cmap="gray", norm=LogNorm(vmin=log_mag_cropped.min() + 1e-6,
                                                          vmax=log_mag_cropped.max()))
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================
# MODE 2: Denoising via FFT
# ============================

def run_mode_2(image_path: str, cutoff_fraction: float = 0.1):
    """
    Mode 2:
    - Load image
    - Pad to powers of two
    - Compute 2D FFT
    - Zero out high frequencies (simple low-pass filter in frequency domain)
    - Inverse FFT
    - Show original + denoised
    - Print number of non-zero coefficients and fraction of total
    cutoff_fraction: fraction of min(rows, cols) used as low-frequency window size.
    """
    img = load_image_gray(image_path)
    padded, orig_shape = pad_to_power_of_two_2d(img)
    rows, cols = padded.shape

    F = fft_2d(padded)
    F_shift = fft_shift_2d(F)

    # Build low-pass mask: keep central square region
    cy, cx = rows // 2, cols // 2
    half_h = int(cutoff_fraction * rows / 2)
    half_w = int(cutoff_fraction * cols / 2)

    mask = np.zeros_like(F_shift, dtype=bool)
    y_start = max(cy - half_h, 0)
    y_end = min(cy + half_h, rows)
    x_start = max(cx - half_w, 0)
    x_end = min(cx + half_w, cols)
    mask[y_start:y_end, x_start:x_end] = True

    F_filtered_shift = np.zeros_like(F_shift)
    F_filtered_shift[mask] = F_shift[mask]

    # Count non-zero coefficients
    nonzeros = np.count_nonzero(F_filtered_shift)
    total = F_filtered_shift.size
    fraction = nonzeros / total if total > 0 else 0.0

    print(f"[Denoise] Non-zero coefficients: {nonzeros} / {total} "
          f"({fraction * 100:.4f}% of original)")

    # Shift back and inverse FFT
    F_filtered = np.fft.ifftshift(F_filtered_shift)
    denoised_padded = ifft_2d(F_filtered)
    denoised = crop_to_original_2d(denoised_padded, orig_shape)
    denoised = to_displayable_image(denoised)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Denoised image")
    plt.imshow(denoised, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================
# MODE 3: Compression via FFT coefficients
# ============================

def run_mode_3(image_path: str):
    """
    Mode 3:
    - Load image
    - Pad to powers of two
    - Compute 2D FFT once
    - Create 6 compression levels, from 0% (original) to very high compression
    - For each level:
        * Zero out smallest coefficients by magnitude
        * Inverse FFT
        * Show in 2x3 subplot
        * Print number of non-zeros
    """
    img = load_image_gray(image_path)
    padded, orig_shape = pad_to_power_of_two_2d(img)
    F = fft_2d(padded)

    total = F.size
    magnitudes = np.abs(F).flatten()
    # Sort magnitudes ascending
    sorted_mags = np.sort(magnitudes)

    # Define compression levels (fraction of coefficients set to zero)
    zero_fracs = [0.0, 0.9, 0.99, 0.999, 0.9995, 0.9999]

    reconstructed_images = []
    nonzero_counts = []

    for frac_zero in zero_fracs:
        if frac_zero <= 0.0:
            # No compression
            mask = np.ones_like(F, dtype=bool)
        else:
            num_zero = int(frac_zero * total)
            num_keep = total - num_zero
            if num_keep <= 0:
                # All zero
                threshold = np.inf
            else:
                # Keep the largest num_keep coefficients
                threshold = sorted_mags[-num_keep]

            mask = np.abs(F) >= threshold

        F_compressed = np.zeros_like(F)
        F_compressed[mask] = F[mask]

        nonzeros = np.count_nonzero(F_compressed)
        nonzero_counts.append(nonzeros)
        print(f"[Compression] Zero fraction: {frac_zero:.4f}, "
              f"non-zero coeffs: {nonzeros} / {total} "
              f"({nonzeros / total * 100:.4f}% kept)")

        img_rec_padded = ifft_2d(F_compressed)
        img_rec = crop_to_original_2d(img_rec_padded, orig_shape)
        img_rec = to_displayable_image(img_rec)
        reconstructed_images.append((frac_zero, img_rec))

    # Plot 2x3 subplot
    plt.figure(figsize=(12, 8))

    for idx, (frac_zero, rec_img) in enumerate(reconstructed_images):
        plt.subplot(2, 3, idx + 1)
        if frac_zero == 0.0:
            title = "0% compression\n(0% zeros)"
        else:
            title = f"{frac_zero * 100:.2f}% zeros"
        plt.title(title)
        plt.imshow(rec_img, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================
# MODE 4: Runtime plots
# ============================

def run_mode_4():
    """
    Mode 4:
    - Create 2D random arrays of size 2^k x 2^k
    - k from MIN_POWER to MAX_POWER
    - For each size, run naïve 2D DFT and 2D FFT NUM_TRIALS times
    - Compute mean and variance of runtimes
    - Plot both curves with error bars (std dev)
    """
    sizes = [2 ** k for k in range(MIN_POWER, MAX_POWER + 1)]

    naive_means = []
    naive_vars = []
    fft_means = []
    fft_vars = []

    for N in sizes:
        print(f"\n[Runtime] Testing size {N} x {N} ...")
        naive_times = []
        fft_times = []

        for t in range(NUM_TRIALS):
            arr = np.random.rand(N, N)

            # Time naive 2D DFT
            start = time.perf_counter()
            dft_2d(arr)
            end = time.perf_counter()
            naive_times.append(end - start)

            # Time 2D FFT
            start = time.perf_counter()
            # pad_to_power_of_two_2d is unnecessary because N is already a power of 2
            fft_2d(arr)
            end = time.perf_counter()
            fft_times.append(end - start)

        naive_times = np.array(naive_times)
        fft_times = np.array(fft_times)

        naive_means.append(naive_times.mean())
        naive_vars.append(naive_times.var())
        fft_means.append(fft_times.mean())
        fft_vars.append(fft_times.var())

        print(f"Naive DFT - mean: {naive_means[-1]:.6f}s, var: {naive_vars[-1]:.6e}")
        print(f"FFT       - mean: {fft_means[-1]:.6f}s, var: {fft_vars[-1]:.6e}")

    naive_means = np.array(naive_means)
    naive_std = np.sqrt(naive_vars)
    fft_means = np.array(fft_means)
    fft_std = np.sqrt(fft_vars)

    # Plot runtimes
    plt.figure(figsize=(8, 6))
    plt.errorbar(sizes, naive_means, yerr=naive_std, marker="o",
                 linestyle="-", label="Naïve 2D DFT")
    plt.errorbar(sizes, fft_means, yerr=fft_std, marker="s",
                 linestyle="-", label="2D FFT")

    plt.xlabel("Problem size (N x N)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime comparison: Naïve 2D DFT vs 2D FFT")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # Print summary in console as well
    print("\n=== Runtime Summary ===")
    for N, nm, nv, fm, fv in zip(sizes, naive_means, naive_vars, fft_means, fft_vars):
        print(f"N={N:4d} | Naive mean={nm:.6f}s var={nv:.6e} | FFT mean={fm:.6f}s var={fv:.6e}")


# ============================
# Argument parsing & main
# ============================

def parse_args():
    parser = argparse.ArgumentParser(description="FFT assignment program")
    parser.add_argument(
        "-m",
        "--mode",
        type=int,
        default=1,
        help="Mode: [1]=FFT display, [2]=denoise, [3]=compress, [4]=runtime plots",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default=DEFAULT_IMAGE,
        help="Path to input image file (for modes 1-3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode
    image_path = args.image

    if mode in [1, 2, 3] and image_path is None:
        raise ValueError("Image path must be provided for modes 1, 2, and 3.")

    if mode == 1:
        print(f"Running mode 1: FFT visualization on image '{image_path}'")
        run_mode_1(image_path)
    elif mode == 2:
        print(f"Running mode 2: Denoising via FFT on image '{image_path}'")
        run_mode_2(image_path)
    elif mode == 3:
        print(f"Running mode 3: Compression via FFT on image '{image_path}'")
        run_mode_3(image_path)
    elif mode == 4:
        print("Running mode 4: Runtime plots")
        run_mode_4()
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
