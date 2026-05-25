"""Gradio presentation layer for Smart Product Intelligence."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import predict_product_insights  # noqa: E402
from src.utils import ProjectConfig, ensure_directories, set_global_seed  # noqa: E402


config = ProjectConfig()


def generate_placeholder_insights(review_text: str, product_title: str, rating: float) -> dict[str, str | float]:
    """Return placeholder product intelligence output for the Gradio demo."""

    metadata = {
        "product_title": product_title,
        "rating": rating,
        "category": config.category,
    }
    return predict_product_insights(model=None, review_text=review_text, metadata=metadata)


def build_app() -> gr.Blocks:
    """Build the Gradio UI."""

    with gr.Blocks(title="Smart Product Intelligence") as demo:
        gr.Markdown("# Smart Product Intelligence")
        gr.Markdown("Placeholder capstone interface for Amazon Reviews 2023 All_Beauty analysis.")

        with gr.Row():
            product_title = gr.Textbox(label="Product Title", placeholder="Example: Hydrating facial cleanser")
            rating = gr.Slider(label="Rating", minimum=1, maximum=5, step=0.5, value=4.5)

        review_text = gr.Textbox(
            label="Review Text",
            lines=5,
            placeholder="Paste a beauty product review here.",
        )
        analyze_button = gr.Button("Generate Placeholder Insights", variant="primary")
        output = gr.JSON(label="Product Intelligence Output")

        analyze_button.click(
            fn=generate_placeholder_insights,
            inputs=[review_text, product_title, rating],
            outputs=output,
        )

    return demo


def main() -> None:
    """Launch the Gradio app."""

    ensure_directories()
    set_global_seed(config.random_seed)
    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    main()

