# Smart Product Intelligence

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers%20%7C%20Diffusers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-App-F97316?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-Amazon%20Reviews%202023-2563EB?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Capstone%20Portfolio-16A34A?style=for-the-badge)

## Overview

**Smart Product Intelligence** is an AI engineering capstone project that analyzes beauty product reviews and metadata from the `McAuley-Lab/Amazon-Reviews-2023` dataset. It is focused on the `All_Beauty` category and demonstrates a complete multimodal product intelligence workflow across tabular modeling, computer vision, NLP, transformers, retrieval-augmented generation, diffusion image generation, and an integrated Gradio application.

This repository is designed as a professional portfolio project for GitHub, LinkedIn, internship applications, and technical interviews. It highlights practical machine learning engineering decisions: robust data preparation, product-level splitting to reduce leakage, modular notebooks, safe app fallbacks, and clear final reporting.

## Demo

The final application is implemented in [app/app.py](app/app.py) with five Gradio tabs:

| Tab | Capability | Source Milestone |
| --- | --- | --- |
| Predicted Rating | Predict product rating band from tabular metadata | Milestone 1 |
| Image Analysis | Upload product image and return category-style analysis | Milestone 2 |
| Similar Products | Search similar products from text query | Milestone 3 |
| Review Intelligence | Extract pros, cons, summary, and grounded QA | Milestone 5 |
| AI Image Generation | Generate or safely mock product hero images | Milestone 6 |

The app is built to keep running even if trained model artifacts are unavailable. When artifacts are missing, it returns transparent fallback outputs instead of crashing.

## Key Features

- 🧠 **Rating prediction** with Logistic Regression and Keras MLP workflows.
- 🖼️ **Image classification pipeline** using a small CNN and MobileNetV2 transfer learning.
- 🔎 **Semantic product search** using TF-IDF cosine similarity.
- ✍️ **Review intelligence** with pros, cons, summaries, and grounded QA.
- 🤖 **Transformer modeling** with DistilBERT and fallback embeddings.
- 🎨 **Diffusion image generation** with Stable Diffusion through `diffusers`.
- 🧩 **Integrated Gradio app** with safe fallbacks for missing artifacts.
- 📊 **EDA and evaluation notebooks** with visualizations, reports, and error analysis.
- 🧱 **Clean architecture** with separate data, model, utility, app, notebook, and report layers.

## AI Technologies Used

| Area | Tools and Models |
| --- | --- |
| Data loading | Hugging Face `datasets`, pandas |
| Tabular ML | scikit-learn, Logistic Regression, feature scaling, imputation |
| Neural networks | TensorFlow, Keras, MLP, TextVectorization, Embedding layers |
| Computer vision | Keras CNN, MobileNetV2 transfer learning |
| NLP search | TF-IDF, cosine similarity |
| Transformers | Hugging Face `transformers`, DistilBERT |
| LLM summarization and QA | `google/flan-t5-small`, `facebook/bart-base`, prompt-template fallback |
| RAG | TF-IDF retrieval index, top-k evidence snippets, grounded response comparison |
| Generative AI images | Hugging Face `diffusers`, Stable Diffusion v1.5, tiny Stable Diffusion fallback |
| Application | Gradio |
| Reporting | Markdown final report and notebook artifacts |

## Architecture Overview

The project follows an MVC-like structure:

| Layer | Path | Responsibility |
| --- | --- | --- |
| Data layer | `src/data.py`, `data/processed/` | Dataset loading, preprocessing, split management |
| Model layer | `src/models.py`, `notebooks/` | Experiments, model training, inference logic |
| Utility layer | `src/utils.py` | Shared paths, configuration, logging, reproducibility |
| Presentation layer | `app/app.py` | Final Gradio user interface |
| Reporting layer | `report/` | Final report, screenshots, presentation artifacts |

## Project Pipeline

```text
Amazon Reviews 2023: All_Beauty
        |
        v
Milestone 0: EDA, cleaning, metadata join, product-level split
        |
        +--> Milestone 1: Tabular rating-band prediction
        |
        +--> Milestone 2: Product image CNN and transfer learning
        |
        +--> Milestone 3: Text classification and similar product search
        |
        +--> Milestone 4: Transformer review classifier
        |
        +--> Milestone 5: LLM summarization and RAG QA
        |
        +--> Milestone 6: Diffusion product image generation
        |
        v
Milestone 7: Integrated Gradio app and final report
```

## Dataset

| Item | Value |
| --- | --- |
| Dataset | `McAuley-Lab/Amazon-Reviews-2023` |
| Category | `All_Beauty` |
| Review config | `raw_review_All_Beauty` |
| Metadata config | `raw_meta_All_Beauty` |
| Processed train split | `data/processed/train.csv` |
| Processed validation split | `data/processed/validation.csv` |
| Processed test split | `data/processed/test.csv` |

Milestone 0 uses a bounded local subset of up to **30,000 reviews** and **10,000 products**. The split is performed by product ID, not by review row, to reduce data leakage between train, validation, and test sets.

## Results

This project produces a full AI product-intelligence workflow rather than a single isolated model. Final numeric scores are generated inside the milestone notebooks after they are run in the target environment.

| Milestone | Result Artifact |
| --- | --- |
| Milestone 0 | Clean train, validation, and test CSV splits |
| Milestone 1 | Baseline vs MLP comparison, classification report, confusion matrix |
| Milestone 2 | Small CNN vs transfer learning comparison and sample predictions |
| Milestone 3 | TF-IDF vs embedding classifier metrics and similar product search |
| Milestone 4 | DistilBERT workflow with TF-IDF baseline and latency comparison |
| Milestone 5 | Product-level summaries, grounded QA, evidence snippets, hallucination analysis |
| Milestone 6 | Diffusion prompts, generated image metadata, image quality reflection |
| Milestone 7 | Integrated Gradio app and final report |

## Screenshots

Add screenshots after running the Gradio app. Placeholder paths are prepared under `report/screenshots/`.

| App View | Screenshot Placeholder |
| --- | --- |
| Predicted Rating | `report/screenshots/01_predicted_rating.png` |
| Image Analysis | `report/screenshots/02_image_analysis.png` |
| Similar Products | `report/screenshots/03_similar_products.png` |
| Review Intelligence | `report/screenshots/04_review_intelligence.png` |
| AI Image Generation | `report/screenshots/05_ai_image_generation.png` |

## Installation Guide

Clone the repository and install dependencies in a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Instructions

Run the final integrated Gradio app:

```bash
python -m app.app
```

The app prints the completion marker when it starts:

```text
Milestone 7 Complete
```

Diffusion image generation is disabled by default in the app to avoid unexpectedly downloading large models. To enable direct diffusion loading:

### Windows

```bash
set SPI_ENABLE_DIFFUSION=1
python -m app.app
```

### macOS or Linux

```bash
export SPI_ENABLE_DIFFUSION=1
python -m app.app
```

## Notebook Descriptions

| Notebook | Purpose |
| --- | --- |
| [00_eda.ipynb](notebooks/00_eda.ipynb) | Loads `raw_review_All_Beauty` and `raw_meta_All_Beauty`, cleans fields, joins metadata, creates EDA plots, and saves product-level splits. |
| [01_tabular_mlp.ipynb](notebooks/01_tabular_mlp.ipynb) | Predicts product rating band from tabular metadata with Logistic Regression and a Keras MLP. |
| [02_vision_cnn_transfer.ipynb](notebooks/02_vision_cnn_transfer.ipynb) | Downloads a manageable image sample, trains a small CNN, and compares it with MobileNetV2 transfer learning. |
| [03_text_embeddings.ipynb](notebooks/03_text_embeddings.ipynb) | Builds TF-IDF and Keras embedding text classifiers and implements similar product search. |
| [04_transformers.ipynb](notebooks/04_transformers.ipynb) | Uses DistilBERT for review rating-band classification with fallback logic for local execution. |
| [05_llm_rag_finetune.ipynb](notebooks/05_llm_rag_finetune.ipynb) | Builds LLM summarization and RAG-based question answering over retrieved review evidence. |
| [06_diffusion.ipynb](notebooks/06_diffusion.ipynb) | Generates product/lifestyle/hero image variations using Stable Diffusion through `diffusers`. |

## Clean Project Structure

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
|   +-- screenshots/
|   |   +-- .gitkeep
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

## Milestone Summary

### Milestone 0: Data Preparation and EDA

- Loads `raw_review_All_Beauty` and `raw_meta_All_Beauty` from Hugging Face.
- Converts reviews and metadata to pandas DataFrames.
- Cleans review text, handles missing values, creates `review_length`, extracts prices where possible, and detects image availability.
- Joins reviews with metadata by product ID, using `parent_asin` when available.
- Saves product-level train, validation, and test splits.

### Milestone 1: Tabular MLP

- Builds tabular features from price, rating count, review length, image availability, and helpful votes when available.
- Trains a Logistic Regression baseline.
- Trains a Keras MLP with ReLU layers, dropout, and softmax output.
- Reports learning curves, confusion matrix, classification report, and error analysis.

### Milestone 2: Vision CNN and Transfer Learning

- Detects image URL columns safely.
- Downloads and validates local images under `data/images/`.
- Trains a small CNN from scratch.
- Trains a MobileNetV2 transfer learning model.
- Compares accuracy, macro-F1, confusion matrix, and sample predictions.

### Milestone 3: Text Embeddings and Search

- Detects review text and rating columns safely.
- Recreates `rating_band` from ratings every run.
- Compares TF-IDF plus Logistic Regression against a Keras embedding model.
- Implements TF-IDF cosine similarity search for similar products.

### Milestone 4: Transformers

- Fine-tunes or safely falls back from DistilBERT (`distilbert-base-uncased`) for review rating-band classification.
- Compares transformer performance against a TF-IDF baseline.
- Includes classification report, confusion matrix, latency comparison, and error analysis.

### Milestone 5: LLM Summarization and RAG

- Aggregates reviews at product level.
- Creates review-derived pros, cons, and summaries.
- Uses `google/flan-t5-small`, falls back to `facebook/bart-base`, then deterministic prompt templates if needed.
- Builds a TF-IDF retrieval index and returns grounded QA with evidence snippets.
- Includes hallucination-risk analysis.

### Milestone 6: Diffusion Image Generation

- Builds prompts from product title, description, and category.
- Uses `runwayml/stable-diffusion-v1-5` as the preferred model.
- Falls back to `hf-internal-testing/tiny-stable-diffusion-pipe`.
- Saves generated images and metadata under `data/generated/`.
- Discusses quality, failure modes, and augmentation potential for Milestone 2.

### Milestone 7: Integrated Application and Report

- Builds the final five-tab Gradio application.
- Adds safe fallbacks if trained artifacts are unavailable.
- Creates [report/final_report.md](report/final_report.md) with the final capstone report structure.
- Prints `Milestone 7 Complete` when the app starts.

## Challenges

- **Schema variability:** Amazon review and metadata fields may appear under different names, so notebooks detect columns defensively.
- **Data leakage risk:** Reviews for the same product can leak across splits unless the split is performed by product ID.
- **Runtime constraints:** Vision, transformer, LLM, and diffusion workflows can be expensive, so notebooks use `FAST_MODE` and bounded samples.
- **Missing artifacts:** The final app must remain usable even when trained model files are not available locally.
- **Generative reliability:** LLM and diffusion outputs require grounding, validation, and clear communication of uncertainty.

## Lessons Learned

- Clean data contracts matter as much as model complexity.
- Product-level splitting is essential for realistic evaluation.
- Baselines make neural and transformer results easier to interpret.
- RAG improves trust by connecting generated answers to retrieved evidence.
- Diffusion outputs are useful for ideation, but not reliable product truth.
- A portfolio-ready AI project needs reproducible notebooks, a clear app surface, and honest limitations.

## Future Improvements

- Save trained Milestone 1 and Milestone 2 artifacts and load them directly in the Gradio app.
- Add experiment tracking with MLflow, Weights & Biases, or TensorBoard summaries.
- Add automated notebook execution checks in CI.
- Expand RAG with vector databases such as FAISS or Chroma.
- Add model cards for each trained model.
- Add real screenshots and a short demo video.
- Improve app styling and deploy the Gradio interface.
- Add stronger evaluation for hallucination, fairness, and product claim safety.

## Acknowledgements

- McAuley Lab for the Amazon Reviews 2023 dataset.
- Hugging Face for `datasets`, `transformers`, and `diffusers`.
- TensorFlow and Keras for deep learning workflows.
- scikit-learn for baselines, preprocessing, metrics, and TF-IDF search.
- Gradio for rapid ML application prototyping.

## Report

The final report scaffold is available at [report/final_report.md](report/final_report.md). It includes:

- Introduction
- Dataset
- Milestones 0 through 6
- Integration
- Results
- Limitations
- Conclusion
- References
