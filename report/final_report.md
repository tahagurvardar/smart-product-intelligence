# Smart Product Intelligence Final Report

## Introduction

Smart Product Intelligence is a capstone project that explores product understanding for the Amazon Reviews 2023 `All_Beauty` category. The project combines tabular metadata, images, review text, transformer models, retrieval-augmented generation, and diffusion-based visual generation into a single application-oriented workflow.

The goal is not to ship a production recommendation system, but to demonstrate an end-to-end machine learning architecture that can support product rating prediction, image analysis, semantic search, review intelligence, and AI-assisted marketing imagery.

## Dataset

- Source: `McAuley-Lab/Amazon-Reviews-2023`
- Category: `All_Beauty`
- Review config: `raw_review_All_Beauty`
- Metadata config: `raw_meta_All_Beauty`

The processed project data is saved as:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

The data split is product-based rather than review-based to reduce leakage between train, validation, and test sets.

## Milestone 0

Milestone 0 created the exploratory data analysis and preprocessing notebook. It loaded reviews and product metadata from Hugging Face, selected manageable local subsets, cleaned important fields, detected image availability, created review-length features, joined review and metadata tables, and saved product-level train, validation, and test splits.

Key output:

- `notebooks/00_eda.ipynb`
- Processed split CSV files under `data/processed/`

## Milestone 1

Milestone 1 focused on tabular rating-band prediction. It built a Logistic Regression baseline and a Keras MLP using metadata-derived features such as price, rating count, review length, image availability, and helpful votes when present.

Key output:

- `notebooks/01_tabular_mlp.ipynb`
- Baseline vs MLP comparison
- Classification metrics, confusion matrix, learning curves, and error analysis

## Milestone 2

Milestone 2 introduced product image modeling. It detected image URL columns safely, downloaded a bounded local cache of images, and compared a small CNN with a MobileNetV2 transfer-learning model.

Key output:

- `notebooks/02_vision_cnn_transfer.ipynb`
- Cached images under `data/images/`
- Accuracy, macro-F1, confusion matrix, sample predictions, and model comparison table

## Milestone 3

Milestone 3 used review text for classification and semantic product search. It compared TF-IDF plus Logistic Regression with a simple Keras embedding classifier. It also built a TF-IDF cosine similarity search workflow for product discovery.

Key output:

- `notebooks/03_text_embeddings.ipynb`
- Text classification metrics
- Similar product search demo for queries such as `gentle moisturizer for sensitive skin`

## Milestone 4

Milestone 4 replaced the simple embedding classifier with a transformer-based workflow using DistilBERT. It kept execution bounded with `FAST_MODE` and included a fallback strategy using frozen transformer embeddings plus Logistic Regression if fine-tuning fails locally.

Key output:

- `notebooks/04_transformers.ipynb`
- TF-IDF baseline comparison
- Transformer evaluation metrics and latency comparison

## Milestone 5

Milestone 5 added LLM summarization and retrieval-augmented question answering. It aggregated reviews at the product level, generated pros, cons, and short summaries, built a TF-IDF retrieval index, and compared grounded and non-grounded answers.

Key output:

- `notebooks/05_llm_rag_finetune.ipynb`
- Product-level review summaries
- RAG answer with confidence score and retrieved evidence snippets
- Hallucination-risk analysis

## Milestone 6

Milestone 6 explored diffusion-based product imagery. It generated prompts from product title, description, and category metadata, then configured Stable Diffusion through `diffusers` with a tiny fallback model for local testing.

Key output:

- `notebooks/06_diffusion.ipynb`
- Generated image output path: `data/generated/`
- Prompt metadata, generation time, image comparison display, and quality reflection

## Integration

Milestone 7 integrates prior milestones into a Gradio application in `app/app.py`.

Application tabs:

- Predicted Rating: tabular rating-band prediction interface for Milestone 1
- Image Analysis: product image upload interface for Milestone 2
- Similar Products: semantic product search from Milestone 3
- Review Intelligence: pros, cons, summary, and grounded QA from Milestone 5
- AI Image Generation: hero image generation workflow from Milestone 6

The application includes safe fallbacks when trained model artifacts are not available. These fallbacks keep the demo usable while clearly reporting when outputs are heuristic or placeholder-based.

## Results

The project produces a complete experimentation and integration structure:

- Clean processed data splits
- Tabular baseline and MLP workflows
- Image CNN and transfer-learning workflows
- Text classification and search workflows
- Transformer review classifier workflow
- LLM review intelligence and RAG workflow
- Diffusion prompt and image generation workflow
- Integrated Gradio app

Final quantitative results should be copied from each executed notebook after running the notebooks in the target environment.

## Limitations

- Some workflows are intentionally bounded by `FAST_MODE`, so notebook results may not represent full-dataset performance.
- Model artifacts are not guaranteed to exist until notebooks are executed and saved intentionally.
- Diffusion image generation can be slow on CPU and may require Hugging Face model access.
- Generated product images can invent packaging, labels, or claims and should not be treated as verified product assets.
- Review-derived summaries and RAG answers depend on retrieved evidence quality and can miss minority opinions.
- Product-level splitting reduces leakage risk, but additional validation is needed before production use.

## Conclusion

Smart Product Intelligence demonstrates a full capstone pipeline for product analytics across structured metadata, images, review text, transformers, LLMs, RAG, and diffusion generation. The final Gradio app shows how these separate milestone workflows can be presented as a single product intelligence interface with safe fallbacks.

## References

- McAuley Lab. `McAuley-Lab/Amazon-Reviews-2023`.
- TensorFlow and Keras documentation.
- Hugging Face `datasets`, `transformers`, and `diffusers` documentation.
- Gradio documentation.
- scikit-learn documentation.
- MobileNetV2 and DistilBERT model references.
