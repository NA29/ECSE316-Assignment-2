#!/usr/bin/env python3
import argparse
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DEFAULT_IMAGE = "moonlanding.png"
MIN_POWER = 5          # 2^5 = 32
MAX_POWER = 7          # 2^7 = 128
NUM_TRIALS = 10        # number of repetitions per size



# 1D DFT (naïve, O(N^2))
def dft_1d(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    X = np.zeros(N, dtype=np.complex128)

    for k in range(N):
        n = np.arange(N)
        exponent = -2j * np.pi * k * n / N
        X[k] = np.sum(x * np.exp(exponent))

    return X



# 1D FFT (Cooley–Tukey, O(N log N))
def fft_1d(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]

    if N == 1:
        return x.copy()

    if N % 2 != 0:
        raise ValueError(f"fft_1d length must be a power of 2 (got N = {N}).")

    X_even = fft_1d(x[0::2])
    X_odd = fft_1d(x[1::2])

    k = np.arange(N // 2)
    twiddle = np.exp(-2j * np.pi * k / N) * X_odd

    X = np.zeros(N, dtype=np.complex128)
    X[:N // 2] = X_even + twiddle
    X[N // 2:] = X_even - twiddle

    return X


# 1D Inverse FFT
def ifft_1d(X):
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]

    X_conj = np.conjugate(X)
    y = fft_1d(X_conj)
    x = np.conjugate(y) / N

    return x



# power-of-two padding
def _next_power_of_two(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 << (n - 1).bit_length()


def pad_to_power_of_two_2d(arr):
    arr = np.asarray(arr)
    rows, cols = arr.shape

    new_rows = _next_power_of_two(rows)
    new_cols = _next_power_of_two(cols)

    if new_rows == rows and new_cols == cols:
        return arr.astype(np.complex128), (rows, cols)

    padded = np.zeros((new_rows, new_cols), dtype=np.complex128)
    padded[:rows, :cols] = arr

    return padded, (rows, cols)


def crop_to_original_2d(arr, original_shape):
    rows, cols = original_shape
    return arr[:rows, :cols]



# 2D FFT / IFFT
def fft_2d(mat):
    A = np.asarray(mat, dtype=np.complex128)
    rows, cols = A.shape

    temp = np.zeros_like(A, dtype=np.complex128)
    for i in range(rows):
        temp[i, :] = fft_1d(A[i, :])

    F = np.zeros_like(temp, dtype=np.complex128)
    for j in range(cols):
        F[:, j] = fft_1d(temp[:, j])

    return F


def ifft_2d(F):
    F = np.asarray(F, dtype=np.complex128)
    rows, cols = F.shape

    temp = np.zeros_like(F, dtype=np.complex128)
    for j in range(cols):
        temp[:, j] = ifft_1d(F[:, j])

    A = np.zeros_like(temp, dtype=np.complex128)
    for i in range(rows):
        A[i, :] = ifft_1d(temp[i, :])

    return A



# naïve 2D DFT (for runtime comparison)
def dft_2d(mat):
    A = np.asarray(mat, dtype=np.complex128)
    rows, cols = A.shape

    temp = np.zeros_like(A, dtype=np.complex128)
    for i in range(rows):
        temp[i, :] = dft_1d(A[i, :])

    F = np.zeros_like(temp, dtype=np.complex128)
    for j in range(cols):
        F[:, j] = dft_1d(temp[:, j])

    return F


# utils
def load_image_gray(path):
    """
    Load image and convert to grayscale float in [0,1].
    Uses matplotlib.image.imread to avoid extra dependencies.
    """
    img = plt.imread(path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    if img.ndim == 3:
        img = img[..., :3]
        img = img.mean(axis=2)

    return img


def to_displayable_image(arr):
    arr = np.real(arr)
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < 1e-12:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def fft_magnitude_log(F):
    mag = np.abs(F)
    # avoid log(0)
    return np.log1p(mag)


def fft_shift_2d(F):
    return np.fft.fftshift(F)


# MODE 1: FFT visualization
def run_mode_1(image_path: str):
    img = load_image_gray(image_path)
    padded, orig_shape = pad_to_power_of_two_2d(img)

    F = fft_2d(padded)
    F_shift = fft_shift_2d(F)
    log_mag = fft_magnitude_log(F_shift)

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



# MODE 2: Denoising 
def run_mode_2(image_path: str, cutoff_fraction: float = 0.1):
    img = load_image_gray(image_path)
    padded, orig_shape = pad_to_power_of_two_2d(img)
    rows, cols = padded.shape

    F = fft_2d(padded)
    F_shift = fft_shift_2d(F)

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

    nonzeros = np.count_nonzero(F_filtered_shift)
    total = F_filtered_shift.size
    fraction = nonzeros / total if total > 0 else 0.0

    print(f"[Denoise] Non-zero coefficients: {nonzeros} / {total} "
          f"({fraction * 100:.4f}% of original)")

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


# MODE 3: Compression
def run_mode_3(image_path: str):
    img = load_image_gray(image_path)
    padded, orig_shape = pad_to_power_of_two_2d(img)
    F = fft_2d(padded)

    total = F.size
    magnitudes = np.abs(F).flatten()
    sorted_mags = np.sort(magnitudes)

    zero_fracs = [0.0, 0.9, 0.99, 0.999, 0.9995, 0.9999]

    reconstructed_images = []
    nonzero_counts = []

    for frac_zero in zero_fracs:
        if frac_zero <= 0.0:
            mask = np.ones_like(F, dtype=bool)
        else:
            num_zero = int(frac_zero * total)
            num_keep = total - num_zero
            if num_keep <= 0:
                threshold = np.inf
            else:
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



# MODE 4: Runtime plots
def run_mode_4():
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

            start = time.perf_counter()
            dft_2d(arr)
            end = time.perf_counter()
            naive_times.append(end - start)

            start = time.perf_counter()
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

    print("\n=== Runtime Summary ===")
    for N, nm, nv, fm, fv in zip(sizes, naive_means, naive_vars, fft_means, fft_vars):
        print(f"N={N:4d} | Naive mean={nm:.6f}s var={nv:.6e} | FFT mean={fm:.6f}s var={fv:.6e}")



# parsing 
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
