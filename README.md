# Scaphoid Fracture Detection

This project uses a Faster R-CNN model to detect scaphoid bones and fractures in X-ray images.

## Project Structure

```
├───best_scaphoid_detection.pth     # Pre-trained model for scaphoid detection
├───config                          # Configuration files
├───pre.py                          # Preprocessing script
├───train.py                        # Training script
├───ui.py                           # Main application with Gradio UI
├───fracture_detection              # Data for fracture detection
│   ├───test
│   └───train
├───scaphoid_detection              # Data for scaphoid detection
│   ├───annotations
│   ├───images
│   └───...
├───tools                           # Helper scripts and modules
│   ├───dataloader.py
│   ├───fasterRCNN.py
│   └───...
└───...
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a virtual environment:**
    *   Using `conda`:
        ```bash
        conda create --name scaphoid-env python=3.11
        conda activate scaphoid-env
        ```
    *   Using `venv`:
        ```bash
        python -m venv .venv
        source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
        ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. You can generate one using `pip freeze > requirements.txt` after installing your project's dependencies.)*

## Usage

To run the web interface for fracture detection:

```bash
python ui.py
```

This will launch a Gradio interface in your browser. You can upload an X-ray image to see the scaphoid detection and fracture classification results.

## Training

To train the scaphoid detection model:

```bash
python train.py
```

Make sure your datasets are correctly placed in the `scaphoid_detection` and `fracture_detection` directories. You can modify the training parameters in the `config` directory.
