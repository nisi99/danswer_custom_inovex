import base64
from typing import Any, List

import bs4
import requests
from atlassian import Confluence
from attr import dataclass
from bs4 import SoupStrainer

from requests.exceptions import SSLError

from danswer.file_processing.image_summarization import summarize_image
from danswer.utils.logger import setup_logger

logger = setup_logger()

@dataclass
class PageImage:
    url: str
    title: str
    base64_encoded: str
    summary: str

def _summarize_page_images(
    page: dict[str, Any],
    confluence: Confluence,
    doc_metadata=None
) -> List[PageImage]:
    """tbd"""
    # extract images from page

    if doc_metadata is None:
        doc_metadata = {}
    text = page["body"]["view"]["value"]

    relevant_images = SoupStrainer("img", class_=lambda x: x != 'emoticon')
    soup = bs4.BeautifulSoup(text, "html.parser", parse_only=relevant_images)
    images = soup.find_all()
    images_data: List[PageImage] = []

    page_id = page['id']

    # export each image
    for i, image in enumerate(images):
        image_url = image["src"]

        try:
            # get image from url
            response = confluence.request(path=image_url, absolute=True)
            image_data = response.content

            metadata = doc_metadata

            if 'title' in page:
                metadata["Confluence document title"] = page['title']

            if 'alt' in image:
                metadata["Image HTML alt text"] = image['alt']

            summary = summarize_image(image_data, metadata)

            base64_image = base64.b64encode(image_data).decode("utf-8")

            # save (meta-)data to list for further processing
            images_data.append(
                PageImage(
                    url=image_url,
                    title=f'{page_id}_image_{i}',
                    base64_encoded=base64_image,
                    summary=summary
                )
            )

        except SSLError as e:
            logger.warning(f"SSL Error: {e}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Request Exception: {e}")


    return images_data
