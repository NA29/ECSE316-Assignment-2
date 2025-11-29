# ECSE 316 – Assignment 2 (Group 23)

## Description

This project implements 1D and 2D DFT/FFT routines without using NumPy’s FFT functions.  
The program supports image visualization, denoising, compression, and runtime analysis.

## Included Files

- main.py (all code in one file)
- moonlanding.png (input image)
- Report.pdf
- README.md
- test_fft.py

## Python Version used for writing/testing

3.13.7

## Requirements

```
source venv/bin/activate
pip3 install requirements.txt
```

## How to Run

```
python3 main.py -m <mode> -i moonlanding.png
```

Modes:

- `-m 1` FFT visualization
- `-m 2` Denoising
- `-m 3` Compression
- `-m 4` Runtime comparison

## Notes

- Image is padded to power-of-two size automatically.
- Mode 4 may take a few minutes for larger inputs.
