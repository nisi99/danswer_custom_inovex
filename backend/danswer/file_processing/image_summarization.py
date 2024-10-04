import base64
import os
from io import BytesIO
from typing import Dict

from PIL import Image
from openai import AzureOpenAI

from danswer.utils.logger import setup_logger

logger = setup_logger()


def summarize_image(image_data: bytes, metadata: Dict[str, str] = None) -> str | None:
    """Use ChatGPT to generate a summary of an image."""
    # initialize the Azure OpenAI Model

    if metadata is None:
        metadata = {}

    image_data = _resize_image_if_needed(image_data)

    # encode image (for llm)
    encoded_image = _encode_image(image_data)

    model = AzureOpenAI()

    prompt = """
        Du bist ein Assistent für die Zusammenfassung von Bildern für das Retrieval.
        Fasse den Inhalt des folgenden Bildes zusammen und sei dabei so präzise wie möglich.
        Die Zusammenfassung wird eingebettet und verwendet, um das Originalbild abzurufen.
        Verfasse daher eine prägnante Zusammenfassung des Bildes, die für das Retrieval optimiert ist.
    """

    metadata_joined = "\n".join({f'{k}={v}' for k, v in metadata.items()})

    # build Prompt-Template
    query = [
        {"type": "text", "text": "Fasse den Inhalt und das Motiv des Bildes zusammen."
                                 "Die Seite, auf der dieses Bild zu sehen ist, hat folgende Metadaten:\n"
                                 f"{metadata_joined}"},
        {"type": "image_url", "image_url": {"url": encoded_image}},
    ]

    res = model.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        temperature=0.0
    )
    summary = res.choices[0].message.content

    return summary


deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


def _encode_image(url: bytes) -> str:
    """Getting the base64 string."""
    base64_encoded_data = base64.b64encode(url).decode("utf-8")

    return f"data:image/jpeg;base64,{base64_encoded_data}"


def _resize_image_if_needed(image_data: bytes, max_size_mb: int = 20) -> bytes:
    """Resize image if it's larger than the specified max size in MB."""
    max_size_bytes = max_size_mb * 1024 * 1024

    if len(image_data) > max_size_bytes:
        with Image.open(BytesIO(image_data)) as img:
            logger.warning(f"resizing image...")

            # Reduce dimensions for better size reduction
            img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            output = BytesIO()

            # Save with lower quality for compression
            img.save(output, format='JPEG', quality=85)
            resized_data = output.getvalue()

            return resized_data

    return image_data
