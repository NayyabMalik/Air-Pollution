# Air Pollution Prediction Using Sentinel-2 Imagery

This repository contains a project to predict NO2 pollution levels in Lahore, Pakistan, using Sentinel-2 satellite imagery and machine learning techniques. The project fetches RGB images from Google Earth Engine, preprocesses them with OpenCV, and trains a convolutional neural network (CNN) to predict NO2 concentrations. It includes a visualization of actual vs. predicted NO2 levels. The project is part of my coursework at the National University of Modern Languages, Islamabad, submitted on October 31, 2024, under the supervision of Mam Iqra Nasem.

## Project Overview

The goal is to estimate NO2 pollution levels using Sentinel-2 satellite imagery (RGB bands: B4, B3, B2) for a region in Lahore, Pakistan, from January 1 to January 31, 2025. The project involves:

- **Data Acquisition**: Fetching Sentinel-2 images via Google Earth Engine for a defined region (bounding box: \[74.30, 31.45, 74.40, 31.55\]).
- **Preprocessing**: Converting images to NumPy arrays, resizing to 256x256, and normalizing pixel values.
- **Modeling**: Training a CNN (implemented with TensorFlow) to predict NO2 levels based on imagery.
- **Evaluation**: Visualizing predictions with a scatter plot of actual vs. predicted NO2 levels.
- **Output**: Saving visualizations in `static/images/`.

The project builds on concepts from my deep learning labs, particularly CNN-based classification and regression.

## Dataset

- **Sentinel-2 Imagery**:
  - Source: Google Earth Engine (`COPERNICUS/S2` ImageCollection).
  - Bands: B4 (Red), B3 (Green), B2 (Blue).
  - Region: Lahore, Pakistan (bounding box: \[74.30, 31.45, 74.40, 31.55\]).
  - Date Range: January 1–31, 2025.
  - Resolution: Images resized to 256x256 pixels, normalized to \[0, 1\].
- **NO2 Values**:
  - Assumed to be externally sourced (e.g., ground station measurements or another dataset).
  - Shape: 1D array (`no2_values`) aligned with the number of images.
- **Training Data**:
  - Input (`X`): NumPy array of shape `(n, 256, 256, 3)` for RGB images.
  - Output (`y`): NumPy array of shape `(n, 1)` for NO2 concentrations.
  - Split: 80% training, 20% testing.

## Repository Structure

```
air-pollution-prediction/
├── data/
│   ├── processed_data/               # Processed images and NO2 data (if available)
├── notebooks/
│   ├── Air Pollution.ipynb           # Jupyter notebook for data fetching, preprocessing, modeling, and visualization
├── static/
│   ├── images/
│   │   ├── actual_vs_predicted.png   # Scatter plot of actual vs. predicted NO2 levels
├── requirements.txt                    # Python dependencies
├── LICENSE                            # MIT License
├── README.md                          # This file
```

## Methodology

1. **Data Acquisition**:

   - Use Google Earth Engine to fetch Sentinel-2 images for Lahore, filtered by date and region.
   - Select RGB bands (B4, B3, B2) for visual feature extraction.

2. **Preprocessing**:

   - Download images using `getThumbUrl` with a 1000m scale.
   - Convert to NumPy arrays using OpenCV (`cv2.imread`).
   - Resize images to 256x256 pixels and normalize pixel values to \[0, 1\].
   - Align NO2 values (`no2_values`) with images, ensuring consistent lengths.

3. **Modeling**:

   - Split data into training (80%) and testing (20%) sets using `train_test_split`.
   - Train a CNN model (implied by TensorFlow import) to predict NO2 levels from RGB images.
   - Model architecture: Not fully specified in the notebook but assumed to be a CNN with convolutional and dense layers.

4. **Evaluation and Visualization**:

   - Generate predictions (`y_pred`) for the test set.
   - Create a scatter plot of actual vs. predicted NO2 levels using Matplotlib, with a dashed line for ideal fit.
   - Save the plot to `static/images/actual_vs_predicted.png`.

## Results

- **Data Shapes**:
  - Training: `X_train` (25, 256, 256, 3), `y_train` (25, 1).
  - Testing: `X_test` shape depends on split (e.g., 6–7 samples).
- **Visualization**:
  - Scatter plot (`actual_vs_predicted.png`) shows the correlation between actual and predicted NO2 levels.
  - Red dashed line represents the ideal fit (y=x).
- **Performance**: Accuracy or loss metrics not provided in the notebook; further evaluation (e.g., MSE, R²) is recommended.

## Related Coursework

This project builds on my deep learning labs, particularly:

- **Lab 3: CNN Classification** (`deep-learning-labs/lab_manuals/CNN_Classification.pdf`): CNN fundamentals, relevant to the regression model.
- **Lab 4: CNN Patterns** (`deep-learning-labs/lab_manuals/CNN_Patterns.pdf`): Image preprocessing and feature extraction, applied to Sentinel-2 images.
- **Computer Vision Labs** (`computer-vision-labs/notebooks/`): Image processing techniques (e.g., resizing, normalization) from labs like `image_operations.py`.

See the `deep-learning-labs` and `computer-vision-labs` repositories for details.

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/air-pollution-prediction.git
   cd air-pollution-prediction
   ```

2. **Install Dependencies**:

   Install Python libraries listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Key libraries: `earthengine-api`, `opencv-python`, `numpy`, `matplotlib`, `scikit-learn`, `tensorflow`, `requests`.

3. **Set Up Google Earth Engine**:

   - Install the Earth Engine Python API: `pip install earthengine-api`.
   - Authenticate with GEE:

     ```bash
     earthengine authenticate
     ```
   - Follow the prompts to generate an authentication token.

4. **Prepare Data**:

   - Ensure `no2_values` (NO2 measurements) are available in the `data/` folder or modify the notebook to load them.
   - Processed images are temporarily stored during execution; save them in `data/processed_data/` if needed.

5. **Run the Notebook**:

   Launch Jupyter Notebook and execute the analysis:

   ```bash
   jupyter notebook notebooks/Air Pollution.ipynb
   ```

6. **View Visualizations**:

   - The scatter plot is saved in `static/images/actual_vs_predicted.png`.

## Usage

1. **Authenticate with Google Earth Engine**:

   - Run `ee.Authenticate()` and `ee.Initialize()` to access Sentinel-2 data.

2. **Fetch and Preprocess Data**:

   - Define the region (Lahore: \[74.30, 31.45, 74.40, 31.55\]) and date range (Jan 2025).
   - Fetch Sentinel-2 images and convert to 256x256 NumPy arrays.

3. **Train and Evaluate Model**:

   - Split data into training and testing sets.
   - Train a CNN model to predict NO2 levels (modify the notebook to include the model definition if needed).
   - Generate predictions and evaluate performance.

4. **Visualize Results**:

   - Create a scatter plot of actual vs. predicted NO2 levels.
   - Save the plot to `static/images/`.

**Example** (Preprocessing and Visualization):

```python
import ee
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ee.Authenticate()
ee.Initialize()

# Define region and date range
region = ee.Geometry.Rectangle([74.30, 31.45, 74.40, 31.55])
start_date = '2025-01-01'
end_date = '2025-01-31'

# Fetch Sentinel-2 images
sentinel2_dataset = ee.ImageCollection("COPERNICUS/S2") \
    .filterDate(start_date, end_date) \
    .filterBounds(region) \
    .select(['B4', 'B3', 'B2'])

# Convert to NumPy array (simplified)
def get_sentinel2_array(image):
    url = image.getThumbUrl({'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2'], 'scale': 1000})
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(response.content)
            img = cv2.imread(temp_file.name)
            if img is not None:
                return cv2.resize(img, (256, 256))
    return None

# Assume no2_values is available
X = [get_sentinel2_array(ee.Image(sentinel2_dataset.toList(sentinel2_dataset.size()).get(i))) for i in range(sentinel2_dataset.size().getInfo())]
X = np.array([x for x in X if x is not None], dtype=np.float32) / 255.0
y = np.array(no2_values[:len(X)]).reshape(-1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plot (y_pred assumed from model)
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle="dashed")
plt.xlabel("Actual NO2 Levels")
plt.ylabel("Predicted NO2 Levels")
plt.legend()
plt.title("Actual vs. Predicted NO2 Pollution")
plt.savefig("static/images/actual_vs_predicted.png")
plt.show()
```

## Future Improvements

- **Model Definition**: Include the CNN architecture (e.g., convolutional layers, pooling, dense layers) in the notebook for clarity.
- **Evaluation Metrics**: Add metrics like Mean Squared Error (MSE) or R² to quantify model performance.
- **NO2 Data Source**: Specify the source of `no2_values` and include it in the repository or provide a script to fetch it.
- **Feature Engineering**: Incorporate additional Sentinel-2 bands (e.g., aerosols, NIR) or meteorological data to improve predictions.
- **Web Interface**: Develop a Flask-based interface (similar to `sales-forecasting/app.py`) for interactive NO2 prediction.
- **Real-Time Monitoring**: Extend to real-time pollution monitoring using GEE streaming.

## Notes

- **File Size**: Use Git LFS for large files (e.g., `git lfs track "*.png" "*.csv"`).
- **Data Limitation**: The notebook assumes `no2_values` is available; ensure it’s included or sourced appropriately.
- **GEE Authentication**: Requires a Google Earth Engine account and authentication token.

## License

This repository is licensed under the MIT License. See the LICENSE file for details.
