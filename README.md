# Smart Product Intelligence

Capstone project scaffold for a Smart Product Intelligence system using Python, TensorFlow/Keras, Hugging Face datasets/models, and Gradio.

The project is designed around the `McAuley-Lab/Amazon-Reviews-2023` dataset, focused on the `All_Beauty` category.

## Project Goals

- Load and prepare Amazon Reviews 2023 beauty product review data.
- Build placeholder architecture for future TensorFlow/Keras and Hugging Face models.
- Provide a Gradio interface for product intelligence experiments.
- Keep the codebase organized with an MVC-like clean architecture:
  - `src/data.py`: data access and preprocessing layer.
  - `src/models.py`: model construction and inference layer.
  - `src/utils.py`: shared configuration, paths, logging, and helpers.
  - `app/app.py`: Gradio presentation layer.

## Project Structure

```text
smart-product-intelligence/
+-- app/
|   +-- __init__.py
|   +-- app.py
+-- data/
|   +-- images/
|   |   +-- .gitkeep
|   +-- processed/
|   |   +-- .gitkeep
|   |   +-- text_search/
|   |   |   +-- .gitkeep
+-- notebooks/
|   +-- 00_eda.ipynb
|   +-- 01_tabular_mlp.ipynb
|   +-- 02_vision_cnn_transfer.ipynb
|   +-- 03_text_embeddings.ipynb
|   +-- README.md
+-- report/
|   +-- README.md
+-- src/
|   +-- __init__.py
|   +-- data.py
|   +-- models.py
|   +-- utils.py
+-- .gitignore
+-- README.md
+-- requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Gradio App

```bash
python -m app.app
```

The initial app is a placeholder interface. Full model loading, training, and inference should be implemented later.

## Dataset

Target dataset:

- Hugging Face dataset: `McAuley-Lab/Amazon-Reviews-2023`
- Category: `All_Beauty`

Planned dataset responsibilities live in `src/data.py`.

## Milestone 0

Milestone 0 prepares an exploratory data analysis and preprocessing workflow for the `All_Beauty` category from `McAuley-Lab/Amazon-Reviews-2023`.

Prepared in [notebooks/00_eda.ipynb](notebooks/00_eda.ipynb):

- Loads `raw_review_All_Beauty` and `raw_meta_All_Beauty` from Hugging Face.
- Uses bounded local subsets of up to 30,000 reviews and 10,000 products.
- Converts reviews and metadata to pandas DataFrames.
- Cleans review text, handles missing values, creates `review_length`, extracts numeric prices where possible, and detects image availability.
- Joins reviews with metadata by product ID, using `parent_asin` when available.
- Splits by product ID into train, validation, and test sets to reduce data leakage risk.

Processed split files are written by the notebook to:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

## Milestone 1

Milestone 1 trains and evaluates first tabular models for product rating-band prediction.

Prepared in [notebooks/01_tabular_mlp.ipynb](notebooks/01_tabular_mlp.ipynb):

- Loads the Milestone 0 processed splits from `data/processed/train.csv`, `data/processed/validation.csv`, and `data/processed/test.csv`.
- Aggregates review-level rows to product-level rows by `product_id`.
- Builds tabular features from price, rating count, mean review length, image availability, and helpful votes when available.
- Creates a product rating-band target, preferring product `average_rating` and falling back to mean `review_rating` when metadata is sparse.
- Handles missing values with median imputation and scales numeric features.
- Trains a Logistic Regression baseline.
- Trains a Keras MLP with ReLU hidden layers, dropout, and a softmax output layer.
- Compares baseline and MLP performance with learning curves, confusion matrices, classification reports, and error analysis.

## Milestone 2

Milestone 2 trains and evaluates image-based models from cached product/review image URLs.

Prepared in [notebooks/02_vision_cnn_transfer.ipynb](notebooks/02_vision_cnn_transfer.ipynb):

- Loads the Milestone 0 processed splits from `data/processed/train.csv`, `data/processed/validation.csv`, and `data/processed/test.csv`.
- Detects image-like columns safely and extracts nested image URLs from strings, lists, dictionaries, and JSON-like values.
- Downloads and validates a bounded local image cache under `data/images/`.
- Uses up to 1,000 train images, 300 validation images, and 300 test images for manageable local execution.
- Builds labels from reliable category signals when available, otherwise falls back to the product rating-band target.
- Trains a small CNN from scratch.
- Trains a MobileNetV2 transfer learning model.
- Compares models with training curves, accuracy, macro-F1, confusion matrices, classification reports, sample predictions, and a comparison table.

## Milestone 3

Milestone 3 trains text-based rating-band models and builds semantic product search.

Prepared in [notebooks/03_text_embeddings.ipynb](notebooks/03_text_embeddings.ipynb):

- Loads the Milestone 0 processed splits from `data/processed/train.csv`, `data/processed/validation.csv`, and `data/processed/test.csv`.
- Detects the review text column safely and creates rating-band labels when needed.
- Trains a TF-IDF Vectorizer plus Logistic Regression baseline.
- Trains a Keras text embedding neural network with `TextVectorization`, an embedding layer, global average pooling, dense layers, dropout, and softmax output.
- Compares text models with accuracy, macro-F1, classification reports, confusion matrices, and error examples.
- Builds product-level search text from product titles, descriptions, feature bullets, and review summaries where available.
- Generates semantic embeddings with `sentence-transformers` when available, with a TF-IDF fallback.
- Implements cosine similarity search and demonstrates a query for `gentle moisturizer for sensitive skin`.
- Saves reusable search artifacts under `data/processed/text_search/` when the notebook is run.

Milestone 4 is not implemented yet.

## Development Notes

This repository intentionally does not implement full models yet. It provides clean extension points for:

- Dataset loading and sampling.
- Text preprocessing.
- TensorFlow/Keras model creation.
- Hugging Face tokenizer/model integration.
- Gradio demo workflows.
- Report and notebook artifacts.
