import base64
import logging
import os
from io import BytesIO

from openai import AzureOpenAI
from openai import RateLimitError
from PIL import Image
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential

from danswer.utils.logger import setup_logger

logger = setup_logger()


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger.logger, logging.WARN),
)
def summarize_image(image_data: bytes, query: str | None = None) -> str | None:
    """Use ChatGPT to generate a summary of an image."""
    # initialize the Azure OpenAI Model

    image_data = _resize_image_if_needed(image_data)

    # encode image (for llm)
    encoded_image = _encode_image(image_data)

    model = AzureOpenAI()

    if not query:
        query = "Summarize the content and the subject of the picture."

    res = model.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": """
                    You are an assistant for summarizing images for retrieval.
                    Summarize the content of the following image and be as precise as possible.
                    The summary will be embedded and used to retrieve the original image.
                    Therefore, write a concise summary of the image that is optimized for retrieval.
                """,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": encoded_image}},
                ],
            },
        ],
        temperature=0.0,
    )
    summary = res.choices[0].message.content

    return summary


deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


def _encode_image(image_data: bytes) -> str:
    """Getting the base64 string."""
    base64_encoded_data = base64.b64encode(image_data).decode("utf-8")

    return f"data:image/jpeg;base64,{base64_encoded_data}"


def _resize_image_if_needed(image_data: bytes, max_size_mb: int = 20) -> bytes:
    """Resize image if it's larger than the specified max size in MB."""
    max_size_bytes = max_size_mb * 1024 * 1024

    if len(image_data) > max_size_bytes:
        with Image.open(BytesIO(image_data)) as img:
            logger.info("resizing image...")

            # Reduce dimensions for better size reduction
            img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            output = BytesIO()

            # Save with lower quality for compression
            img.save(output, format="JPEG", quality=85)
            resized_data = output.getvalue()

            return resized_data

    return image_data
