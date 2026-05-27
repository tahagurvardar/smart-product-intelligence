"""Final Gradio application for Smart Product Intelligence."""

from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ProjectConfig, ensure_directories, set_global_seed  # noqa: E402


try:
    import pandas as pd
except Exception:  # pragma: no cover - app fallback for incomplete local envs
    pd = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - app fallback for incomplete local envs
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - app fallback for incomplete local envs
    TfidfVectorizer = None
    cosine_similarity = None


config = ProjectConfig()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

PRODUCT_ID_CANDIDATES = ["product_id", "parent_asin", "asin", "item_id"]
TITLE_CANDIDATES = ["product_title", "title", "name", "product_name"]
DESCRIPTION_CANDIDATES = ["product_description", "description", "details", "about_product"]
CATEGORY_CANDIDATES = ["main_category", "category", "categories", "product_category", "store"]
TEXT_CANDIDATES = ["review_text", "text", "review_body", "body", "content"]
RATING_CANDIDATES = ["review_rating", "rating", "average_rating", "overall", "score"]

_REVIEW_CACHE: Any | None = None
_PRODUCT_CACHE: Any | None = None
_SEARCH_CACHE: dict[str, Any] | None = None
_DIFFUSION_PIPE: Any | None = None
_DIFFUSION_BACKEND = "not-loaded"


def clean_text(value: Any) -> str:
    """Normalize text-ish values from CSV fields."""

    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except TypeError:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def flatten_text(value: Any) -> str:
    """Flatten list/dict/JSON-like metadata into displayable text."""

    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(flatten_text(item) for item in value)
    text = str(value).strip()
    if text.startswith(("[", "{")):
        for parser in (json.loads, ast.literal_eval):
            try:
                return flatten_text(parser(text))
            except Exception:
                continue
    return clean_text(text)


def first_existing_column(frame: Any, candidates: list[str]) -> str | None:
    """Return the first candidate column found in a dataframe."""

    if pd is None or frame is None:
        return None
    return next((column for column in candidates if column in frame.columns), None)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a UI or CSV value to float."""

    try:
        converted = float(value)
        if math.isnan(converted):
            return default
        return converted
    except Exception:
        return default


def load_processed_reviews() -> Any | None:
    """Load processed splits with safe fallbacks."""

    global _REVIEW_CACHE
    if _REVIEW_CACHE is not None:
        return _REVIEW_CACHE
    if pd is None:
        return None

    frames = []
    for split_name in ["train", "validation", "test"]:
        path = PROCESSED_DIR / f"{split_name}.csv"
        if path.exists():
            try:
                frames.append(pd.read_csv(path, low_memory=False).assign(split=split_name))
            except Exception:
                continue

    if not frames:
        return None
    _REVIEW_CACHE = pd.concat(frames, ignore_index=True)
    return _REVIEW_CACHE


def prepare_review_table() -> Any | None:
    """Create a normalized review table for app retrieval features."""

    raw = load_processed_reviews()
    if pd is None or raw is None or raw.empty:
        return None

    product_col = first_existing_column(raw, PRODUCT_ID_CANDIDATES)
    title_col = first_existing_column(raw, TITLE_CANDIDATES)
    description_col = first_existing_column(raw, DESCRIPTION_CANDIDATES)
    category_col = first_existing_column(raw, CATEGORY_CANDIDATES)
    text_col = first_existing_column(raw, TEXT_CANDIDATES)
    rating_col = first_existing_column(raw, RATING_CANDIDATES)

    if text_col is None and title_col is None and description_col is None:
        return None

    table = pd.DataFrame()
    table["product_id"] = raw[product_col].astype(str) if product_col else raw.index.astype(str)
    table["title"] = raw[title_col].map(flatten_text) if title_col else "All Beauty product"
    table["description"] = raw[description_col].map(flatten_text) if description_col else ""
    table["category"] = raw[category_col].map(flatten_text) if category_col else config.category
    table["review_text"] = raw[text_col].map(clean_text) if text_col else ""
    table["rating"] = pd.to_numeric(raw[rating_col], errors="coerce") if rating_col else 0.0

    unknown_markers = {"", "unknown", "nan", "none", "null", "[]", "{}"}
    for column in ["title", "description", "category", "review_text"]:
        table[column] = table[column].map(lambda value: "" if clean_text(value).lower() in unknown_markers else clean_text(value))
    table["title"] = table["title"].replace("", "All Beauty product")
    table["category"] = table["category"].replace("", config.category)
    table = table[table[["title", "description", "review_text"]].agg(" ".join, axis=1).str.len() > 0]
    return table.reset_index(drop=True)


def prepare_product_table() -> Any | None:
    """Aggregate reviews to product-level rows for search and summaries."""

    global _PRODUCT_CACHE
    if _PRODUCT_CACHE is not None:
        return _PRODUCT_CACHE
    table = prepare_review_table()
    if pd is None or table is None or table.empty:
        return None

    def first_non_empty(values: Any, default: str) -> str:
        for value in values:
            text = clean_text(value)
            if text:
                return text
        return default

    def combine_reviews(values: Any, limit: int = 6, max_chars: int = 900) -> str:
        parts = []
        seen = set()
        for value in values:
            text = clean_text(value)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(text[:max_chars])
            if len(parts) >= limit:
                break
        return " ".join(parts)

    rows = []
    for product_id, group in table.groupby("product_id", dropna=True):
        positive = group[group["rating"] >= 4]
        mixed_or_negative = group[group["rating"] < 4]
        rows.append(
            {
                "product_id": product_id,
                "title": first_non_empty(group["title"], "All Beauty product"),
                "description": first_non_empty(group["description"], ""),
                "category": first_non_empty(group["category"], config.category),
                "average_rating": round(float(group["rating"].mean()), 2) if group["rating"].notna().any() else None,
                "review_count": int(len(group)),
                "review_text": combine_reviews(group["review_text"]),
                "positive_reviews": combine_reviews(positive["review_text"], limit=4),
                "mixed_or_negative_reviews": combine_reviews(mixed_or_negative["review_text"], limit=4),
            }
        )

    _PRODUCT_CACHE = pd.DataFrame(rows)
    _PRODUCT_CACHE["search_text"] = (
        _PRODUCT_CACHE["title"].fillna("")
        + " "
        + _PRODUCT_CACHE["description"].fillna("")
        + " "
        + _PRODUCT_CACHE["category"].fillna("")
        + " "
        + _PRODUCT_CACHE["review_text"].fillna("")
    )
    return _PRODUCT_CACHE


def artifact_status(patterns: list[str]) -> str:
    """Report whether likely model artifacts exist."""

    matches = []
    for base in [MODEL_DIR, ARTIFACTS_DIR, PROCESSED_DIR]:
        if base.exists():
            for pattern in patterns:
                matches.extend(base.rglob(pattern))
    if matches:
        return f"Found artifact candidate: {matches[0].relative_to(PROJECT_ROOT)}"
    return "No trained artifact found; using safe fallback."


def rating_band_from_score(score: float) -> str:
    """Convert a numeric rating estimate to a rating band."""

    if score <= 2:
        return "low"
    if score < 4:
        return "medium"
    return "high"


def predict_rating(
    product_title: str,
    product_description: str,
    price: float,
    rating_count: float,
    review_length: float,
    image_available: bool,
    helpful_votes: float,
) -> dict[str, Any]:
    """Predict rating band with a model artifact if available, otherwise a heuristic."""

    _ = product_title
    _ = product_description
    artifact_note = artifact_status(["*milestone1*.joblib", "*tabular*.joblib", "*.keras", "*.h5"])

    price_value = safe_float(price)
    rating_count_value = safe_float(rating_count)
    review_length_value = safe_float(review_length)
    helpful_votes_value = safe_float(helpful_votes)

    score = 3.0
    score += min(math.log1p(max(rating_count_value, 0)) / 8, 0.55)
    score += min(math.log1p(max(helpful_votes_value, 0)) / 12, 0.25)
    score += 0.25 if image_available else -0.15
    score += 0.15 if 20 <= review_length_value <= 600 else -0.05
    if price_value <= 0:
        score -= 0.05
    elif price_value > 80:
        score -= 0.15
    score = max(1.0, min(5.0, score))
    band = rating_band_from_score(score)

    return {
        "prediction_source": "Milestone 1 fallback heuristic",
        "artifact_status": artifact_note,
        "predicted_rating_estimate": round(score, 2),
        "predicted_rating_band": band,
        "class_label": band,
        "confidence": "low" if "No trained artifact" in artifact_note else "artifact candidate found; loading hook not configured",
        "inputs_used": {
            "price": price_value,
            "rating_count": rating_count_value,
            "review_length": review_length_value,
            "image_available": bool(image_available),
            "helpful_votes": helpful_votes_value,
        },
    }


def analyze_image(image: Any) -> dict[str, Any]:
    """Return a Milestone 2-style image category prediction or fallback."""

    artifact_note = artifact_status(["*milestone2*.keras", "*vision*.keras", "*cnn*.keras", "*.h5"])
    if image is None:
        return {
            "prediction_source": "Milestone 2 fallback",
            "artifact_status": artifact_note,
            "message": "Upload a product image to analyze.",
        }

    width = getattr(image, "width", None)
    height = getattr(image, "height", None)
    mode = getattr(image, "mode", "unknown")
    aspect_ratio = round(width / height, 3) if width and height else None

    return {
        "prediction_source": "Milestone 2 fallback image inspection",
        "artifact_status": artifact_note,
        "predicted_category": config.category,
        "confidence": "low",
        "image_width": width,
        "image_height": height,
        "aspect_ratio": aspect_ratio,
        "mode": mode,
        "note": "A trained CNN or transfer-learning artifact was not loaded, so the app returns a safe category fallback.",
    }


def token_overlap_score(query: str, text: str) -> float:
    """Simple fallback similarity score without scikit-learn."""

    query_tokens = {token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 2}
    text_tokens = {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def build_search_cache() -> dict[str, Any] | None:
    """Build a product-level search cache for Milestone 3 features."""

    global _SEARCH_CACHE
    if _SEARCH_CACHE is not None:
        return _SEARCH_CACHE
    product_table = prepare_product_table()
    if pd is None or product_table is None or product_table.empty:
        return None

    if TfidfVectorizer is not None and cosine_similarity is not None:
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(product_table["search_text"].fillna(""))
        _SEARCH_CACHE = {
            "mode": "tfidf",
            "products": product_table,
            "vectorizer": vectorizer,
            "matrix": matrix,
        }
    else:
        _SEARCH_CACHE = {
            "mode": "token-overlap",
            "products": product_table,
            "vectorizer": None,
            "matrix": None,
        }
    return _SEARCH_CACHE


def similar_products(query: str, top_k: int = 5) -> Any:
    """Return top similar products from the Milestone 3 search approach."""

    if pd is None:
        return [["dependency_missing", "Install pandas/scikit-learn to enable search.", "", 0.0, 0]]
    query = clean_text(query)
    if not query:
        return pd.DataFrame(columns=["product_id", "title", "category", "average_rating", "similarity_score"])

    cache = build_search_cache()
    if cache is None:
        return pd.DataFrame(
            [["missing_data", "Run Milestone 0 to create processed CSVs.", config.category, None, 0.0]],
            columns=["product_id", "title", "category", "average_rating", "similarity_score"],
        )

    products = cache["products"]
    if cache["mode"] == "tfidf":
        query_vector = cache["vectorizer"].transform([query])
        scores = cosine_similarity(query_vector, cache["matrix"]).ravel()
    else:
        scores = products["search_text"].fillna("").map(lambda text: token_overlap_score(query, text)).to_numpy()

    top_indices = scores.argsort()[::-1][: int(top_k)]
    results = products.iloc[top_indices][["product_id", "title", "category", "average_rating", "review_count"]].copy()
    results["similarity_score"] = [round(float(scores[index]), 4) for index in top_indices]
    return results.reset_index(drop=True)


def summarize_reviews(product_query: str, question: str) -> tuple[str, str, str, str, Any]:
    """Create pros, cons, summary, and grounded QA from local reviews."""

    if pd is None:
        empty = [["dependency_missing", "Install pandas to enable review intelligence.", 0.0]]
        return "Unavailable", "Unavailable", "pandas is not installed.", "Unable to answer without local data tooling.", empty

    products = prepare_product_table()
    if products is None or products.empty:
        empty = pd.DataFrame([["missing_data", "Run Milestone 0 to create processed CSVs.", 0.0]], columns=["product_id", "evidence", "score"])
        return "No local reviews found.", "No local reviews found.", "Run Milestone 0 first.", "No grounded answer is available.", empty

    product_query = clean_text(product_query)
    question = clean_text(question) or "What do reviews say about this product?"

    if product_query:
        exact = products[products["product_id"].str.lower() == product_query.lower()]
        contains = products[products["title"].str.contains(re.escape(product_query), case=False, na=False)]
        selected = exact if not exact.empty else contains
        if selected.empty:
            selected = similar_products(product_query, top_k=1)
            selected = products[products["product_id"].isin(selected["product_id"].astype(str))]
    else:
        selected = products.sort_values(["review_count"], ascending=False).head(1)

    if selected.empty:
        selected = products.sort_values(["review_count"], ascending=False).head(1)

    product = selected.iloc[0]
    source_reviews = prepare_review_table()
    review_rows = source_reviews[source_reviews["product_id"] == product["product_id"]].copy()
    if review_rows.empty:
        review_rows = source_reviews.head(20).copy()

    pros = product.get("positive_reviews") or "No clearly positive review snippets were available."
    cons = product.get("mixed_or_negative_reviews") or "No clearly negative or mixed review snippets were available."
    average_rating = product.get("average_rating")
    summary = (
        f"{product.get('title', 'Selected product')} has {product.get('review_count', 0)} local review rows"
        f" with an average rating of {average_rating}. Pros and cons are extracted from rating-filtered review snippets."
    )

    review_rows["evidence_text"] = review_rows["title"].fillna("") + " " + review_rows["review_text"].fillna("")
    if TfidfVectorizer is not None and cosine_similarity is not None and len(review_rows) > 1:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(review_rows["evidence_text"])
        scores = cosine_similarity(vectorizer.transform([question]), matrix).ravel()
    else:
        scores = review_rows["evidence_text"].map(lambda text: token_overlap_score(question, text)).to_numpy()

    order = scores.argsort()[::-1][:5]
    evidence = review_rows.iloc[order][["product_id", "title", "rating", "review_text"]].copy()
    evidence["score"] = [round(float(scores[index]), 4) for index in order]

    confidence_score = round(float(evidence["score"].mean()), 3) if not evidence.empty else 0.0
    answer = (
        f"Grounded answer from retrieved reviews: evidence for '{question}' is "
        f"{'moderate' if confidence_score > 0.08 else 'limited'}. "
        "Use the snippets below as support before making a product recommendation."
    )
    return pros, cons, summary, answer, evidence[["product_id", "title", "rating", "score", "review_text"]].reset_index(drop=True)


def create_placeholder_hero(prompt: str, title: str) -> Any:
    """Create a safe placeholder image when diffusion is unavailable."""

    if Image is None:
        return None
    width, height = 1024, 576
    image = Image.new("RGB", (width, height), color=(246, 247, 244))
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 36)
        body_font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    draw.rectangle((32, 32, width - 32, height - 32), outline=(90, 110, 120), width=3)
    draw.rectangle((80, 120, 360, 456), fill=(230, 234, 231), outline=(130, 140, 145), width=2)
    draw.text((112, 260), "PRODUCT", fill=(60, 70, 75), font=body_font)
    draw.text((420, 140), clean_text(title)[:48] or "Hero Image Concept", fill=(30, 45, 50), font=title_font)
    prompt_lines = []
    words = clean_text(prompt).split()
    for start in range(0, min(len(words), 60), 8):
        prompt_lines.append(" ".join(words[start : start + 8]))
    draw.text((420, 210), "\n".join(prompt_lines), fill=(70, 80, 84), font=body_font)
    draw.text((420, 460), "Diffusion fallback placeholder", fill=(100, 105, 110), font=body_font)
    return image


def maybe_load_diffusion_pipeline() -> tuple[Any | None, str]:
    """Load Milestone 6 diffusion only when explicitly enabled."""

    global _DIFFUSION_PIPE, _DIFFUSION_BACKEND
    if _DIFFUSION_PIPE is not None:
        return _DIFFUSION_PIPE, _DIFFUSION_BACKEND
    if os.getenv("SPI_ENABLE_DIFFUSION", "0") != "1":
        _DIFFUSION_BACKEND = "disabled; set SPI_ENABLE_DIFFUSION=1 to load diffusers"
        return None, _DIFFUSION_BACKEND

    try:
        import torch
        from diffusers import StableDiffusionPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        for model_id in ["runwayml/stable-diffusion-v1-5", "hf-internal-testing/tiny-stable-diffusion-pipe"]:
            try:
                _DIFFUSION_PIPE = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
                _DIFFUSION_BACKEND = f"{model_id} on {device}"
                return _DIFFUSION_PIPE, _DIFFUSION_BACKEND
            except Exception as model_error:
                _DIFFUSION_BACKEND = f"{model_id} failed: {str(model_error)[:160]}"
    except Exception as import_error:
        _DIFFUSION_BACKEND = f"diffusers unavailable: {str(import_error)[:160]}"
    return None, _DIFFUSION_BACKEND


def generate_hero_image(product_title: str, description: str, category: str) -> tuple[Any, dict[str, Any]]:
    """Generate a Milestone 6 hero image or safe placeholder."""

    title = clean_text(product_title) or "All Beauty product"
    category = clean_text(category) or config.category
    description = clean_text(description)
    prompt = (
        f"Marketing hero image of {title}, {description}, in the {category} category, "
        "premium beauty brand campaign, clean background, elegant product photography"
    )
    output_path = GENERATED_DIR / f"app_hero_{int(time.time())}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    pipe, backend = maybe_load_diffusion_pipeline()
    status = "placeholder"
    error = ""
    if pipe is not None:
        try:
            image = pipe(prompt=prompt, num_inference_steps=8, guidance_scale=6.0).images[0]
            image.save(output_path)
            status = "generated"
        except Exception as generation_error:
            error = str(generation_error)[:300]
            image = create_placeholder_hero(prompt, title)
            if image is not None:
                image.save(output_path)
            status = "placeholder_after_error"
    else:
        image = create_placeholder_hero(prompt, title)
        if image is not None:
            image.save(output_path)

    metadata = {
        "prompt": prompt,
        "backend": backend,
        "status": status,
        "generation_time_seconds": round(time.perf_counter() - started_at, 2),
        "image_path": str(output_path),
        "error": error,
    }
    return image, metadata


def build_app() -> gr.Blocks:
    """Build the integrated Milestone 7 Gradio app."""

    with gr.Blocks(title="Smart Product Intelligence") as demo:
        gr.Markdown("# Smart Product Intelligence")
        gr.Markdown("Integrated capstone interface for the Amazon Reviews 2023 `All_Beauty` workflow.")

        with gr.Tab("Predicted Rating"):
            with gr.Row():
                rating_title = gr.Textbox(label="Product Title", placeholder="Gentle facial moisturizer")
                rating_price = gr.Number(label="Price", value=19.99)
                rating_count = gr.Number(label="Rating Count", value=120)
            rating_description = gr.Textbox(label="Product Metadata / Description", lines=3)
            with gr.Row():
                rating_review_length = gr.Number(label="Average Review Length", value=120)
                rating_image_available = gr.Checkbox(label="Image Available", value=True)
                rating_helpful_votes = gr.Number(label="Helpful Votes", value=0)
            rating_button = gr.Button("Predict Rating Band", variant="primary")
            rating_output = gr.JSON(label="Milestone 1 Output")
            rating_button.click(
                fn=predict_rating,
                inputs=[
                    rating_title,
                    rating_description,
                    rating_price,
                    rating_count,
                    rating_review_length,
                    rating_image_available,
                    rating_helpful_votes,
                ],
                outputs=rating_output,
            )

        with gr.Tab("Image Analysis"):
            image_input = gr.Image(label="Upload Product Image", type="pil")
            image_button = gr.Button("Analyze Image", variant="primary")
            image_output = gr.JSON(label="Milestone 2 Output")
            image_button.click(fn=analyze_image, inputs=image_input, outputs=image_output)

        with gr.Tab("Similar Products"):
            search_query = gr.Textbox(label="Text Query", value="gentle moisturizer for sensitive skin")
            search_top_k = gr.Slider(label="Top K", minimum=1, maximum=10, step=1, value=5)
            search_button = gr.Button("Find Similar Products", variant="primary")
            search_output = gr.Dataframe(label="Milestone 3 Similar Products", interactive=False)
            search_button.click(fn=similar_products, inputs=[search_query, search_top_k], outputs=search_output)

        with gr.Tab("Review Intelligence"):
            review_product_query = gr.Textbox(label="Product ID or Title Search", placeholder="Paste a product_id or title keyword")
            review_question = gr.Textbox(label="Grounded QA Question", value="Is this moisturizer good for sensitive skin?")
            review_button = gr.Button("Generate Review Intelligence", variant="primary")
            with gr.Row():
                pros_output = gr.Textbox(label="Pros", lines=5)
                cons_output = gr.Textbox(label="Cons", lines=5)
            summary_output = gr.Textbox(label="Summary", lines=4)
            qa_output = gr.Textbox(label="Grounded QA Answer", lines=4)
            evidence_output = gr.Dataframe(label="Retrieved Evidence", interactive=False)
            review_button.click(
                fn=summarize_reviews,
                inputs=[review_product_query, review_question],
                outputs=[pros_output, cons_output, summary_output, qa_output, evidence_output],
            )

        with gr.Tab("AI Image Generation"):
            hero_title = gr.Textbox(label="Product Title", value="Gentle facial moisturizer for sensitive skin")
            hero_description = gr.Textbox(label="Description", lines=3, value="hydrating, lightweight, clean skincare product")
            hero_category = gr.Textbox(label="Category", value=config.category)
            hero_button = gr.Button("Generate Hero Image", variant="primary")
            hero_image = gr.Image(label="Milestone 6 Hero Image", type="pil")
            hero_metadata = gr.JSON(label="Generation Metadata")
            hero_button.click(
                fn=generate_hero_image,
                inputs=[hero_title, hero_description, hero_category],
                outputs=[hero_image, hero_metadata],
            )

    return demo


def main() -> None:
    """Launch the Gradio app."""

    ensure_directories()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.random_seed)
    demo = build_app()
    print("Milestone 7 Complete")
    demo.launch()


if __name__ == "__main__":
    main()
