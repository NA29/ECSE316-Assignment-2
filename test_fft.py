import numpy as np
import fft

def test_basic_dft():
    # simple check
    data = np.array([1, 0, 0, 0], dtype=complex)
    res = fft.dft_1d(data)
    expected = np.ones(4, dtype=complex)
    assert np.allclose(res, expected), "basic dft failed"

def test_fft_vs_dft():
    # compare fft and dft
    rng = np.random.default_rng(0)
    sizes = [2, 4, 8, 16, 32]
    for n in sizes:
        sig = rng.random(n) + 1j * rng.random(n)
        dft_res = fft.dft_1d(sig)
        fft_res = fft.fft_1d(sig)
        
        if not np.allclose(fft_res, dft_res):
            print(f"Mismatch at N={n}")
            raise Exception("fft != dft")

def test_inverse_1d():
    # check if ifft(fft(x)) == x
    rng = np.random.default_rng(1)
    for n in [2, 4, 8, 16, 32]:
        orig = rng.random(n) + 1j * rng.random(n)
        transformed = fft.fft_1d(orig)
        recon = fft.ifft_1d(transformed)
        
        assert np.allclose(recon, orig), f"Inverse failed for N={n}"

def test_padding():
    # make sure padding and cropping works
    rng = np.random.default_rng(2)
    shapes = [(30, 50), (17, 64), (100, 101)]
    for s in shapes:
        arr = rng.random(s)
        padded, old_shape = fft.pad_to_power_of_two_2d(arr)
        back = fft.crop_to_original_2d(padded, old_shape)
        
        if not np.array_equal(arr, back):
            raise Exception(f"Padding roundtrip failed for {s}")

def test_2d_inverse():
    # 2d ifft check
    rng = np.random.default_rng(3)
    for n in [4, 8, 16, 32]:
        mat = rng.random((n, n))
        f = fft.fft_2d(mat)
        rec = fft.ifft_2d(f)
        
        assert np.allclose(rec, mat), f"2D inverse failed N={n}"

def test_numpy_compare():
    # compare with numpy
    import numpy.fft as nf
    rng = np.random.default_rng(4)
    
    for s in [(30, 50), (32, 64)]:
        a = rng.random(s)
        pad, _ = fft.pad_to_power_of_two_2d(a)
        
        my_fft = fft.fft_2d(pad)
        np_fft = nf.fft2(pad)
        
        assert np.allclose(my_fft, np_fft), f"Numpy mismatch {pad.shape}"

def test_energy():
    # parsevalish check
    rng = np.random.default_rng(5)
    a = rng.random((16, 16))
    f = fft.fft_2d(a)
    
    n_a = np.linalg.norm(a)
    n_f = np.linalg.norm(f)
    
    ratio = n_f / n_a
    # just check it's not crazy
    if ratio < 0.1 or ratio > 100:
        print(f"Energy ratio weird: {ratio}")
        raise Exception("Energy check failed")

def run_test(test_func):
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    
    try:
        test_func()
        print(f"{GREEN}[PASS]{RESET} {test_func.__name__}")
    except Exception as e:
        print(f"{RED}[FAIL]{RESET} {test_func.__name__}: {e}")

if __name__ == "__main__":
    print("Running tests...")
    run_test(test_basic_dft)
    run_test(test_fft_vs_dft)
    run_test(test_inverse_1d)
    run_test(test_padding)
    run_test(test_2d_inverse)
    run_test(test_numpy_compare)
    run_test(test_energy)
    print("Done.")
