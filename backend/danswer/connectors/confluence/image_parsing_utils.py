import base64
import bs4 # type: ignore
import os

from io import BytesIO
from PIL import Image
from openai import AzureOpenAI                           # type: ignore
#from langchain_openai import AzureOpenAIEmbeddings       # type: ignore


azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT2")
api_version = os.getenv("AZURE_OPENAI_VERSION2")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT2")
deployment_embedding = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS")
api_key = os.getenv("AZURE_OPENAI_API_KEY2")
temperature = os.getenv("TEMPERATURE")


def encode_image(url: str):
    """Getting the base64 string."""
    base64_encoded_data = base64.b64encode(url).decode("utf-8")

    return f"data:image/jpeg;base64,{base64_encoded_data}"


def image_summary(image_base64: str):
    """Use ChatGPT to generate a summary of an image."""
    # initialize the Azure OpenAI Model
    model = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )

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

    # embeddings = AzureOpenAIEmbeddings(
    #     api_version=api_version,
    #     openai_api_type="azure",
    #     azure_endpoint=azure_endpoint,
    #     azure_deployment=deployment_embedding,
    #     api_key=api_key,
    # )

    #return summary, embedding.embed_query(summary)
    return summary


def get_images_data(
    images_data: list,
    soup: bs4.BeautifulSoup,
    confluence,
    page_name: str,
    save_path: str="",
):
    """tbd"""
    # extract images from page
    images = soup.find_all('img')

    # export each image
    for i, image in enumerate(images):
        image_url = image["src"]

        if not image_url.startswith("https"):
            print(f"skipped image with url {image_url} on page {page_name} due to invalid url")

        else:
            # get image from url
            response = confluence.request(path=image_url, absolute=True)
            img = Image.open(BytesIO(response.content))

            # encode image (for llm)
            encoded_image = encode_image(response.content)

            # get summary
            print(f"getting summary of image {i} of page {page_name}")
            #summary, embedding = image_summary(encoded_image)
            summary = image_summary(encoded_image)
            if summary:
                print("done")

            # save image to disc
            if save_path != "":
                img.save(os.path.join(save_path, f'{page_name}_image_{i}.png'))
                print(f'saved image {i} of page: {page_name}')

            # save (meta-)data to list for further processing
            images_data.append(
                {
                    "url": image_url,
                    "title": f'{page_name}_image_{i}',
                    "image": encoded_image,
                    "summary": summary,
                    #"embedding": embedding,
                }
            )

    return images_data
