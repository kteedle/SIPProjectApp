import numpy as np
from PIL import Image
import os

def generate_sample_images():
    """Generate various test images for the pipeline."""
    os.makedirs('sample_data', exist_ok=True)
    
    # 1. 8-bit RGB natural scene (simple gradient pattern)
    rgb_array = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            rgb_array[i, j] = [i, j, (i+j)//2]
    Image.fromarray(rgb_array).save('sample_data/color_sample_1.jpg')
    
    # 2. 8-bit grayscale image
    gray_array = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            gray_array[i, j] = (i ^ j)  # XOR pattern
    Image.fromarray(gray_array).save('sample_data/grayscale_sample_1.png')
    
    # 3. Small image (16x16)
    small_array = np.random.randint(0, 255, (16, 16), dtype=np.uint8)
    Image.fromarray(small_array).save('sample_data/edge_case_small.png')
    
    # 4. Uniform image
    uniform_array = np.full((200, 200), 128, dtype=np.uint8)
    Image.fromarray(uniform_array).save('sample_data/uniform_image.png')
    
    # 5. Noisy image (salt & pepper)
    noisy_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    # Add salt and pepper noise
    salt_pepper = np.random.random((256, 256))
    noisy_array[salt_pepper < 0.05] = 0    # pepper
    noisy_array[salt_pepper > 0.95] = 255  # salt
    Image.fromarray(noisy_array).save('sample_data/noisy_image.png')
    
    print("Sample images generated in sample_data/ folder")

if __name__ == "__main__":
    generate_sample_images()




# """ CURSOR Gen::
# Sample data generation utilities.

# This module contains functions to generate sample data for testing and development purposes.
# """

# import numpy as np
# import pandas as pd
# import os
# from typing import Optional, Tuple


# def generate_random_data(n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
#     """
#     Generate random sample data.
    
#     Args:
#         n_samples: Number of samples to generate
#         n_features: Number of features per sample
        
#     Returns:
#         DataFrame with random data
#     """
#     np.random.seed(42)  # For reproducible results
    
#     data = {
#         f'feature_{i}': np.random.normal(0, 1, n_samples)
#         for i in range(n_features)
#     }
    
#     # Add a target variable
#     data['target'] = np.random.randint(0, 2, n_samples)
    
#     return pd.DataFrame(data)


# def generate_time_series_data(n_points: int = 1000) -> pd.DataFrame:
#     """
#     Generate sample time series data.
    
#     Args:
#         n_points: Number of time points
        
#     Returns:
#         DataFrame with time series data
#     """
#     np.random.seed(42)
    
#     t = np.linspace(0, 10, n_points)
#     signal = np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)
#     noise = np.random.normal(0, 0.1, n_points)
    
#     data = pd.DataFrame({
#         'time': t,
#         'signal': signal + noise,
#         'trend': 0.1 * t,
#         'seasonal': np.sin(2 * np.pi * t)
#     })
    
#     return data


# def generate_classification_data(n_samples: int = 1000, n_classes: int = 3) -> Tuple[pd.DataFrame, np.ndarray]:
#     """
#     Generate sample classification data.
    
#     Args:
#         n_samples: Number of samples
#         n_classes: Number of classes
        
#     Returns:
#         Tuple of (features DataFrame, target array)
#     """
#     from sklearn.datasets import make_classification
    
#     X, y = make_classification(
#         n_samples=n_samples,
#         n_features=4,
#         n_informative=2,
#         n_redundant=0,
#         n_classes=n_classes,
#         random_state=42
#     )
    
#     feature_names = [f'feature_{i}' for i in range(X.shape[1])]
#     X_df = pd.DataFrame(X, columns=feature_names)
    
#     return X_df, y


# def save_sample_data(data: pd.DataFrame, filename: str, output_dir: str = "sample_data") -> str:
#     """
#     Save sample data to file.
    
#     Args:
#         data: DataFrame to save
#         filename: Name of the output file
#         output_dir: Output directory
        
#     Returns:
#         Path to the saved file
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     if filename.endswith('.csv'):
#         filepath = os.path.join(output_dir, filename)
#         data.to_csv(filepath, index=False)
#     elif filename.endswith('.xlsx'):
#         filepath = os.path.join(output_dir, filename)
#         data.to_excel(filepath, index=False)
#     else:
#         filepath = os.path.join(output_dir, f"{filename}.csv")
#         data.to_csv(filepath, index=False)
    
#     print(f"Sample data saved to: {filepath}")
#     return filepath


# def main():
#     """Generate and save sample datasets."""
#     print("Generating sample datasets...")
    
#     # Generate random data
#     random_data = generate_random_data(1000, 10)
#     save_sample_data(random_data, "random_data.csv")
    
#     # Generate time series data
#     ts_data = generate_time_series_data(1000)
#     save_sample_data(ts_data, "time_series_data.csv")
    
#     # Generate classification data
#     X, y = generate_classification_data(1000, 3)
#     X['target'] = y
#     save_sample_data(X, "classification_data.csv")
    
#     print("Sample data generation complete!")


# if __name__ == "__main__":
#     main()
