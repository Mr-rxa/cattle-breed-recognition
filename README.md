# Cattle Breed Recognition

This project uses deep learning to recognize cattle breeds from images. It can also identify if an image is not of a cow or buffalo (e.g., a human or other animal) and labels such cases as **"Unknown"**.

## Features

- Accurate breed classification for cows and buffaloes.
- Handles non-cattle images by predicting "Unknown".
- Detailed evaluation metrics and per-class performance.
- Easy-to-use evaluation script.

## How to Use

1. **Clone the repository:**
   ```
   git clone https://github.com/Mr-rxa/cattle_breed_recognition.git
   cd cattle_breed_recognition
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Prepare your dataset:**
   - Place your processed dataset in the folder specified in `config.py`.
   - Ensure the test images are in `processed_dataset/test/`.

4. **Run evaluation:**
   ```
   python evaluate_model.py
   ```

5. **View results:**
   - Check the console for summary and per-class performance.
   - See `evaluation_results.json` for detailed results.

## SIH Presentation

- The script automatically detects and labels non-cattle images as "Unknown".
- Summary tables and metrics are printed for easy presentation.
- Results are saved for further analysis.

## Project Structure

- `evaluate_model.py` — Evaluation script.
- `config.py` — Configuration settings.
- `requirements.txt` — Python dependencies.
- `evaluation_results.json` — Saved evaluation results.

## Contact

For questions or collaboration, contact [Mr-rxa](https://github.com/Mr-rxa).
