# Image Edge Processing Project

A complete Python application for image edge detection with customizable smoothing and thresholding parameters. Features both GUI and CLI interfaces.

## Features

- **Multiple Smoothing Filters**: Box filter and Gaussian smoothing
- **Sobel Edge Detection**: X, Y, or magnitude gradient computation
- **Flexible Band Processing**: Grayscale conversion or multi-band processing
- **Interactive GUI**: Real-time parameter adjustment and preview
- **Comprehensive Metrics**: Precision, recall, F1-score, Hausdorff distance, and more
- **Multiple Output Formats**: PNG, TIFF with 8-bit or 16-bit depth

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone or download the project**
   ```bash
   cd image_edge_project
   ```
   
2. Create virtual environment
```bash
python3.10 -m venv venv
```

3. Activate virtual environment
* Linux/macOS: source venv/bin/activate
* Windows: venv\Scripts\Activate.ps1

4. Install dependencies
```bash
pip install -r requirements.txt
```

5. Generate sample images (optional)
```bash
python sample_data/generate_samples.py
```
