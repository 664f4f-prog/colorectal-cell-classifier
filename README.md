# Colorectal Cell Classifier

A deep learning project that classifies different cell types in colorectal cancer histology images using transfer learning with EfficientNetB0. This project demonstrates practical applications of computer vision in computational pathology.

## Why this matters

Cell-type classification in histology images is an important step in computational pathology. Automating this process can support research by enabling large-scale, quantitative analysis of tissue composition.

## What this project does

- Loads and preprocesses histology image data from remote sources
- Implements transfer learning using EfficientNetB0 as a base model
- Classifies 8 different colorectal histology cell types:
  - Adipose, Complex, Debris, Empty, Lympho, Mucosa, Stroma, Tumor
- Splits data into training (80%) and testing (20%) sets
- Trains the model for 5 epochs with validation monitoring
- Evaluates classification performance using accuracy metrics
- Visualizes results with training curves and confusion matrices
- Displays sample predictions with confidence scores

## Implementation Details

This project uses **transfer learning** with EfficientNetB0 as the base model. The pre-trained EfficientNetB0 model (trained on ImageNet) is frozen, and custom dense layers are added on top for the 8-class classification task. Images are resized to 224x224 pixels to match the model's input requirements.

The model architecture:
- Base: EfficientNetB0 (frozen, pre-trained on ImageNet)
- Custom layers: Flatten + Dense layer with softmax activation for 8 classes
- Optimizer: Adam
- Loss function: Categorical crossentropy

## Project Structure

```
colorectal-cell-classifier/
├── notebooks/
│   └── colorectal_cell_classification.ipynb  # Jupyter notebook with the full pipeline
├── src/
│   └── classifying_the_different_cells_in_colorectal_cancer.py  # Python script version
├── requirements.txt  # Python dependencies
└── README.md  # This file
```

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

The project requires:
- TensorFlow (for deep learning models)
- TensorFlow Datasets (for data loading)
- NumPy, Pandas (for data manipulation)
- Matplotlib, Seaborn (for visualization)
- Scikit-learn, Scikit-image (for utilities and preprocessing)
- Pillow, Requests (for image handling)

## Running Locally

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd colorectal-cell-classifier
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3.12 -m venv venv  # or python3.11
   source venv/bin/activate  # On macOS/Linux
   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** This project requires Python 3.11 or 3.12 (TensorFlow doesn't support Python 3.14+ yet). If your default `python3` is 3.14+, use `python3.12` or `python3.11` instead.

### Running the Code

The project includes both a Python script and a Jupyter notebook. You can use either:

**Option 1: Python Script**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On macOS/Linux
python src/classifying_the_different_cells_in_colorectal_cancer.py
```

**Option 2: Jupyter Notebook**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On macOS/Linux
jupyter notebook notebooks/colorectal_cell_classification.ipynb
```

### Data Download

On first run, the script will automatically download the required data files (~500MB):
- Images: `data/images.npy`
- Labels: `data/labels.npy`

The data will be saved in a `data/` directory and reused on subsequent runs, so you won't need to download it again. The download may take a few minutes depending on your internet connection.

**Note:** The `data/` directory is excluded from git via `.gitignore` since the files are large. Each developer will download the data on their first run.

## Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **EfficientNetB0** - Pre-trained convolutional neural network for transfer learning
- **NumPy/Pandas** - Data manipulation and preprocessing
- **Matplotlib/Seaborn** - Data visualization and result plotting
- **Scikit-learn** - Machine learning utilities
- **Jupyter Notebooks** - Interactive development and experimentation

## Results

The model achieves strong classification performance on the colorectal histology dataset, successfully distinguishing between 8 different cell types. The transfer learning approach with EfficientNetB0 allows for efficient training while leveraging pre-trained features from ImageNet.

Key metrics include:
- Training accuracy and validation accuracy tracking
- Confusion matrix visualization for detailed performance analysis
- Sample predictions with confidence scores

## License

This project is open source and available for educational purposes.
