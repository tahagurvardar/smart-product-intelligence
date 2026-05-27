# Smart Product Intelligence

Smart Product Intelligence is a capstone project for product analytics on the `McAuley-Lab/Amazon-Reviews-2023` dataset, focused on the `All_Beauty` category.

The project combines Python, pandas, scikit-learn, TensorFlow/Keras, Hugging Face models, diffusion models, and Gradio to explore:

- Product rating-band prediction from tabular metadata.
- Product image analysis.
- Similar product search from review text and metadata.
- Review summarization and grounded question answering.
- AI-assisted product hero image generation.
- A final integrated Gradio application with safe fallbacks when model artifacts are unavailable.

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
|   +-- generated/
|   |   +-- .gitkeep
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
|   +-- 04_transformers.ipynb
|   +-- 05_llm_rag_finetune.ipynb
|   +-- 06_diffusion.ipynb
|   +-- README.md
+-- report/
|   +-- final_report.md
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

## Installation

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

## Run Instructions

Run the final integrated Gradio application:

```bash
python -m app.app
```

The app includes five tabs:

- Predicted Rating
- Image Analysis
- Similar Products
- Review Intelligence
- AI Image Generation

The application is designed to keep running even when trained model artifacts are not present. In that case, it returns clear fallback outputs.

To allow the AI Image Generation tab to load diffusion models directly, set:

```bash
set SPI_ENABLE_DIFFUSION=1
python -m app.app
```

On macOS or Linux:

```bash
export SPI_ENABLE_DIFFUSION=1
python -m app.app
```

## Screenshots

Add final screenshots after launching the Gradio app:

- `Predicted Rating` tab
- `Image Analysis` tab
- `Similar Products` tab
- `Review Intelligence` tab
- `AI Image Generation` tab

Suggested storage location: `report/screenshots/`.

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
- Detects text and rating columns safely and recreates rating-band labels from ratings every run.
- Trains a TF-IDF Vectorizer plus Logistic Regression baseline.
- Trains a Keras text embedding neural network with `TextVectorization`, an embedding layer, global average pooling, dense layers, dropout, and softmax output.
- Compares text models with accuracy, macro-F1, classification reports, and confusion matrices.
- Implements TF-IDF cosine similarity search and demonstrates a query for `gentle moisturizer for sensitive skin`.

## Milestone 4

Milestone 4 replaces the Milestone 3 embedding classifier with a transformer-based review rating-band classifier.

Prepared in [notebooks/04_transformers.ipynb](notebooks/04_transformers.ipynb):

- Loads the Milestone 0 processed splits from `data/processed/train.csv`, `data/processed/validation.csv`, and `data/processed/test.csv`.
- Detects text and rating columns safely and recreates rating-band labels from ratings.
- Uses FAST_MODE limits of up to 3,000 train rows, 1,000 validation rows, and 1,000 test rows.
- Trains a TF-IDF plus Logistic Regression baseline for comparison.
- Fine-tunes DistilBERT (`distilbert-base-uncased`) with Hugging Face transformers when local TensorFlow training works.
- Falls back to frozen DistilBERT embeddings plus Logistic Regression if transformer fine-tuning fails locally.
- Reports accuracy, macro-F1, classification report, confusion matrix, latency comparison, and short error analysis.

## Milestone 5

Milestone 5 adds lightweight LLM review summarization and retrieval-augmented question answering.

Prepared in [notebooks/05_llm_rag_finetune.ipynb](notebooks/05_llm_rag_finetune.ipynb):

- Loads the Milestone 0 processed splits from `data/processed/train.csv`, `data/processed/validation.csv`, and `data/processed/test.csv`.
- Builds product-level review aggregation for summarization.
- Creates review-derived `pros`, `cons`, and `short_summary` fields.
- Uses `google/flan-t5-small` first, falls back to `facebook/bart-base`, and then uses deterministic prompt-template output if local model loading is unavailable.
- Compares zero-shot summaries with a lightweight prompt-template adaptation path.
- Builds a TF-IDF retrieval index over review chunks from the processed data.
- Retrieves the top 5 evidence snippets for `Is this moisturizer good for sensitive skin?`.
- Generates a grounded RAG answer with a confidence score and evidence snippets.
- Compares grounded and non-grounded answers and includes hallucination-risk analysis.

## Milestone 6

Milestone 6 generates alternative product and lifestyle imagery from product metadata using diffusion models.

Prepared in [notebooks/06_diffusion.ipynb](notebooks/06_diffusion.ipynb):

- Loads the Milestone 0 processed splits from `data/processed/train.csv`, `data/processed/validation.csv`, and `data/processed/test.csv`.
- Detects product title, description, category, and product ID columns safely.
- Creates prompts for original product photography, lifestyle imagery, and marketing/hero imagery.
- Configures Stable Diffusion through `diffusers`, with `runwayml/stable-diffusion-v1-5` as the preferred model and `hf-internal-testing/tiny-stable-diffusion-pipe` as the lightweight fallback.
- Uses `FAST_MODE = True` to keep local generation small.
- Saves generated images and generation metadata under `data/generated/`.
- Displays prompt, generation time, and side-by-side image comparisons.
- Includes reflection on visual quality, failure modes, and whether generated images could support Milestone 2 augmentation experiments.

## Milestone 7

Milestone 7 integrates the project into a final application and report structure.

Prepared in [app/app.py](app/app.py) and [report/final_report.md](report/final_report.md):

- Builds a five-tab Gradio application.
- Tab 1 predicts rating band from product metadata using the Milestone 1 interface with safe fallback logic.
- Tab 2 accepts uploaded product images and returns a Milestone 2-style image analysis fallback if no vision artifact is available.
- Tab 3 searches for similar products using the Milestone 3 TF-IDF retrieval workflow when processed data is available.
- Tab 4 returns review pros, cons, summary, grounded QA, and retrieved evidence snippets inspired by Milestone 5.
- Tab 5 generates or safely mocks a Milestone 6 hero image workflow.
- Adds a final report scaffold covering the full capstone pipeline.
- Prints `Milestone 7 Complete` when the app starts.

Future Milestones are not implemented yet.

## Development Notes

This repository intentionally does not implement full models yet. It provides clean extension points for:

- Dataset loading and sampling.
- Text preprocessing.
- TensorFlow/Keras model creation.
- Hugging Face tokenizer/model integration.
- Gradio demo workflows.
- Report and notebook artifacts.
