# Blue Colored Cell Detection

This project implements a YOLOv8-based deep learning model for detecting blue-colored cells in microscopic images. The system includes training, inference, and model export capabilities.

## Project Structure

```
.
├── DATA/                  # Data directory (not included in repo)
│   ├── annotated/        # Annotated images
│   ├── dataset_split/    # Train/validation split data
│   ├── images_tiled/     # Tiled images
│   ├── labels_tiled/     # Labels for tiled images
│   └── wdataset/        # Original dataset
├── exporttflite.py       # Script for exporting model to TFLite
├── requirements.txt      # Python dependencies
├── server.py            # Inference server
├── split_dataset.py     # Dataset splitting utility
├── tiler.py            # Image tiling utility
└── train.py            # Model training script
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Mayuresh2112/Blue-Colored-Cell-Detection.git
cd Blue-Colored-Cell-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset and model files:
   - Create a `DATA` directory in the project root
   - Download and extract the dataset files into `DATA/wdataset/`
   - Download trained models (if needed) into the appropriate directories

## Usage

### Data Preparation

1. Split the dataset into train and validation sets:
```bash
python split_dataset.py
```

2. Tile the images (if needed):
```bash
python tiler.py
```

### Training

To train the model:
```bash
python train.py
```

The training script will:
- Load the dataset configuration from `dataset.yaml`
- Train a YOLOv8 model on the prepared dataset
- Save the best weights and training results in the `runs/detect/train/` directory

### Model Export

To export the trained model to TFLite format:
```bash
python exporttflite.py
```

The script supports exporting to multiple formats:
- TFLite (default)
- ONNX (uncomment the relevant line)
- TensorRT (uncomment the relevant line)

### Inference Server

To run the inference server:
```bash
python server.py
```

The server provides an API endpoint for running inference on new images.

## Model Information

- Base Architecture: YOLOv8
- Input: Microscopic images
- Output: Bounding box coordinates for blue-colored cells
- Training Data: Dataset of annotated microscopic images

## Requirements

Key dependencies include:
- ultralytics
- opencv-python
- numpy
- Flask (for server)

See `requirements.txt` for the complete list of dependencies.

## License

[Add your license information here]

## Acknowledgments

[Add any acknowledgments or references here]
