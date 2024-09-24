import base64
import bs4 # type: ignore
import os
import requests
from sys import getsizeof

from atlassian import Confluence
from requests.exceptions import SSLError
from io import BytesIO
from PIL import Image
from openai import AzureOpenAI                           # type: ignore
#from langchain_openai import AzureOpenAIEmbeddings       # type: ignore
from danswer.utils.logger import setup_logger

logger = setup_logger()

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

def encode_image(url: bytes) -> str:
    """Getting the base64 string."""
    base64_encoded_data = base64.b64encode(url).decode("utf-8")

    return f"data:image/jpeg;base64,{base64_encoded_data}"


def image_summary(image_base64: str) -> str | None:
    """Use ChatGPT to generate a summary of an image."""
    # initialize the Azure OpenAI Model
    model = AzureOpenAI()

    prompt = """
        Du bist ein Assistent für die Zusammenfassung von Bildern für das Retrieval.
        Fasse den Inhalt des folgenden Bildes zusammen und sei dabei so präzise wie möglich.
        Die Zusammenfassung wird eingebettet und verwendet, um das Originalbild abzurufen.
        Verfasse daher eine prägnante Zusammenfassung des Bildes, die für das Retrieval optimiert ist.
    """

    # build Prompt-Template
    res = model.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Fasse den Inhalt und das Motiv des Bildes zusammen."},
                    {"type": "image_url", "image_url": {"url": image_base64}},
                ],
            },
        ],
        temperature=0.0
    )
    summary = res.choices[0].message.content

    return summary


def resize_image_if_needed(image_data: bytes, max_size_mb: int = 20) -> bytes:
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



def get_images_data(
    text: str,
    confluence: Confluence,
    page_name: str,
    save_path: str="None",
) -> list:
    """tbd"""
    # extract images from page
    soup = bs4.BeautifulSoup(text, "html.parser")
    images = soup.find_all('img')
    images_data: list = []

    # export each image
    for i, image in enumerate(images):
        image_url = image["src"]
        logger.info(image_url)

        # if not image_url.startswith("https") or image_url.endswith("ico") or "portal.neusta" in image_url:
        #     logger.info(f"skipped image with url {image_url} on page {page_name} due to invalid url")


        try:
            # get image from url
            response = confluence.request(path=image_url, absolute=True)

            # Resize image if it's larger than 20MB
            image_data = resize_image_if_needed(response.content)

            # encode image (for llm)
            encoded_image = encode_image(image_data)

            # get summary
            logger.info(f"getting summary of image {i} of page {page_name}")
            #summary, embedding = image_summary(encoded_image)
            summary = image_summary(encoded_image)
            if summary:
                logger.info("summary done")

            # save image to disc
            if save_path != "None":
                img = Image.open(BytesIO(response.content))
                img.save(os.path.join(save_path, f'{page_name}_image_{i}.png'))
                logger.info(f'saved image {i} of page: {page_name}')

            # save (meta-)data to list for further processing
            images_data.append(
                {
                    "url": image_url,
                    "title": f'{page_name}_image_{i}',
                    "image": encoded_image,
                    "summary": summary,
                }
            )

        except SSLError as e:
            logger.warning(f"SSL Error: {e}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Request Exception: {e}")


    return images_data
