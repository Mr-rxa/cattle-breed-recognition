A machine learning project for recognizing cattle breeds from images, now featuring a Flask web application. This repository contains code, datasets, and instructions for training, evaluating, and deploying models to classify different cattle breeds.

## Features

- Image classification using deep learning
- Dataset preparation and augmentation
- Model training and evaluation scripts
- Results visualization
- Flask web app for interactive predictions

## Getting Started

1. Clone the repository:
     ```bash
     git clone https://github.com/yourusername/cattle-breed-recognition.git
     ```
2. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
3. Prepare your dataset in the `data/` directory.

4. Train the model:
     ```bash
     python train.py
     ```

## Running the Flask App

Start the web application to make predictions via a browser:
```bash
python app.py
```
Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Dataset

You can use publicly available cattle breed image datasets or collect your own. Place images in `data/train/` and `data/val/` folders, organized by breed.

## Usage

After training, use the model to predict cattle breeds from new images:
```bash
python predict.py --image path/to/image.jpg
```
Or use the Flask app for a web-based interface.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License.
