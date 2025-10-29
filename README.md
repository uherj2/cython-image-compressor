# cython-image-compressor
Optimization of an image compressor project completed in fall of 2024 using cython. The purpose of the original project was to apply numpy and use a kmeans calculation. Using pure python for the project resulted in a slow product.

The purpose of the cython version is to speed up the kmeans process for calcuating colors.  <p>

## Key features
- Separate Optimized and Original Compression

## Requirements

Global requirements: 
- Python 3.9+
- pip, 
- A C compiler <p>

## Step by Step Installation

1. install dependencies: 
`pip install -r requirements.txt`

2. compile cython module:
`pip install -e .`

3. Run files `cyimageCompression.py` and `originalImageCopmression.py` in their respective folders

Note: a comparison script that runs both files is not yet completed.
