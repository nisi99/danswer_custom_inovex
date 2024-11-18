import asyncio
import base64
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from urllib.parse import quote

import bs4  # type: ignore
import requests  # type: ignore
from atlassian import Confluence  # type:ignore
from attr import dataclass  # type: ignore
from bs4 import SoupStrainer  # type: ignore

from danswer.configs.app_configs import CONFLUENCE_CONNECTOR_LABELS_TO_SKIP
from danswer.configs.app_configs import CONTINUE_ON_CONNECTOR_FAILURE
from danswer.configs.app_configs import INDEX_BATCH_SIZE
from danswer.configs.app_configs import MULTIMODAL_ANSWERING_WITH_RAW_IMAGE
from danswer.configs.app_configs import MULTIMODAL_ANSWERING_WITH_SUMMARY_IMAGE
from danswer.configs.chat_configs import SYSTEM_PROMPT
from danswer.configs.chat_configs import USER_PROMPT
from danswer.configs.constants import DocumentSource
from danswer.connectors.confluence.onyx_confluence import handle_confluence_rate_limit
from danswer.connectors.confluence.onyx_confluence import OnyxConfluence
from danswer.connectors.confluence.utils import attachment_to_content
from danswer.connectors.confluence.utils import build_confluence_client
from danswer.connectors.confluence.utils import build_confluence_document_id
from danswer.connectors.confluence.utils import datetime_from_string
from danswer.connectors.confluence.utils import extract_text_from_confluence_html
from danswer.connectors.interfaces import GenerateDocumentsOutput
from danswer.connectors.interfaces import GenerateSlimDocumentOutput
from danswer.connectors.interfaces import LoadConnector
from danswer.connectors.interfaces import PollConnector
from danswer.connectors.interfaces import SecondsSinceUnixEpoch
from danswer.connectors.interfaces import SlimConnector
from danswer.connectors.models import BasicExpertInfo
from danswer.connectors.models import ConnectorMissingCredentialError
from danswer.connectors.models import Document
from danswer.connectors.models import Section
from danswer.connectors.models import SlimDocument
from danswer.file_processing.image_summarization import summarize_image
from danswer.utils.logger import setup_logger

logger = setup_logger()

# Potential Improvements
# 1. Include attachments, etc
# 2. Segment into Sections for more accurate linking, can split by headers but make sure no text/ordering is lost

_COMMENT_EXPANSION_FIELDS = ["body.storage.value"]
_PAGE_EXPANSION_FIELDS = [
    "body.storage.value",
    "version",
    "space",
    "metadata.labels",
]
_ATTACHMENT_EXPANSION_FIELDS = [
    "version",
    "space",
    "metadata.labels",
]

_RESTRICTIONS_EXPANSION_FIELDS = [
    "space",
    "restrictions.read.restrictions.user",
    "restrictions.read.restrictions.group",
]


@dataclass
class ImageSummarization:
    url: str
    title: str
    base64_encoded: str
    media_type: str
    summary: str | None


class ConfluenceConnector(LoadConnector, PollConnector, SlimConnector):
    def __init__(
        self,
        wiki_base: str,
        is_cloud: bool,
        space: str = "",
        page_id: str = "",
        index_recursively: bool = True,
        cql_query: str | None = None,
        batch_size: int = INDEX_BATCH_SIZE,
        continue_on_failure: bool = CONTINUE_ON_CONNECTOR_FAILURE,
        # if a page has one of the labels specified in this list, we will just
        # skip it. This is generally used to avoid indexing extra sensitive
        # pages.
        labels_to_skip: list[str] = CONFLUENCE_CONNECTOR_LABELS_TO_SKIP,
    ) -> None:
        self.batch_size = batch_size
        self.continue_on_failure = continue_on_failure
        self.confluence_client: OnyxConfluence | None = None
        self.is_cloud = is_cloud

        # Remove trailing slash from wiki_base if present
        self.wiki_base = wiki_base.rstrip("/")

        # if nothing is provided, we will fetch all pages
        cql_page_query = "type=page"
        if cql_query:
            # if a cql_query is provided, we will use it to fetch the pages
            cql_page_query = cql_query
        elif space:
            # if no cql_query is provided, we will use the space to fetch the pages
            cql_page_query += f" and space='{quote(space)}'"
        elif page_id:
            if index_recursively:
                cql_page_query += f" and ancestor='{page_id}'"
            else:
                # if neither a space nor a cql_query is provided, we will use the page_id to fetch the page
                cql_page_query += f" and id='{page_id}'"

        self.cql_page_query = cql_page_query
        self.cql_time_filter = ""

        self.cql_label_filter = ""
        if labels_to_skip:
            labels_to_skip = list(set(labels_to_skip))
            comma_separated_labels = ",".join(f"'{label}'" for label in labels_to_skip)
            self.cql_label_filter = f" and label not in ({comma_separated_labels})"

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        # see https://github.com/atlassian-api/atlassian-python-api/blob/master/atlassian/rest_client.py
        # for a list of other hidden constructor args
        self.confluence_client = build_confluence_client(
            credentials_json=credentials,
            is_cloud=self.is_cloud,
            wiki_base=self.wiki_base,
        )
        return None

    def _get_comment_string_for_page_id(self, page_id: str) -> str:
        if self.confluence_client is None:
            raise ConnectorMissingCredentialError("Confluence")

        comment_string = ""

        comment_cql = f"type=comment and container='{page_id}'"
        comment_cql += self.cql_label_filter

        expand = ",".join(_COMMENT_EXPANSION_FIELDS)
        for comments in self.confluence_client.paginated_cql_page_retrieval(
            cql=comment_cql,
            expand=expand,
        ):
            for comment in comments:
                comment_string += "\nComment:\n"
                comment_string += extract_text_from_confluence_html(
                    confluence_client=self.confluence_client,
                    confluence_object=comment,
                )

        return comment_string

    def _convert_object_to_document(
        self, confluence_object: dict[str, Any]
    ) -> Tuple[Optional[Document], Optional[List]]:
        """
        Takes in a confluence object, extracts all metadata, and converts it into a document.
        If its a page, it extracts the text, adds the comments for the document text.
        If its an attachment, it just downloads the attachment and converts that into a document.
        If multimodality is true, images are extracted and summarized by the default LLM.
        """
        if self.confluence_client is None:
            raise ConnectorMissingCredentialError("Confluence")

        # The url and the id are the same
        object_url = build_confluence_document_id(
            self.wiki_base, confluence_object["_links"]["webui"], self.is_cloud
        )

        object_text = None
        # Extract text from page
        if confluence_object["type"] == "page":
            object_text = extract_text_from_confluence_html(
                self.confluence_client, confluence_object
            )
            # Add comments to text
            object_text += self._get_comment_string_for_page_id(confluence_object["id"])
        elif confluence_object["type"] == "attachment":
            object_text = attachment_to_content(
                self.confluence_client, confluence_object
            )

        if object_text is None:
            return None, None

        # Get space name
        doc_metadata: dict[str, str | list[str]] = {
            "Wiki Space Name": confluence_object["space"]["name"]
        }

        # Get labels
        label_dicts = confluence_object["metadata"]["labels"]["results"]
        page_labels = [label["name"] for label in label_dicts]
        if page_labels:
            doc_metadata["labels"] = page_labels

        # Get last modified and author email
        last_modified = datetime_from_string(confluence_object["version"]["when"])
        author_email = confluence_object["version"].get("by", {}).get("email")

        doc = Document(
            id=object_url,
            sections=[Section(link=object_url, text=object_text)],
            source=DocumentSource.CONFLUENCE,
            semantic_identifier=confluence_object["title"],
            doc_updated_at=last_modified,
            primary_owners=(
                [BasicExpertInfo(email=author_email)] if author_email else None
            ),
            metadata=doc_metadata,
        )
        logger.info(
            f"Converted Confluence page - {confluence_object['title']} - to Document"
        )

        image_docs = []
        if MULTIMODAL_ANSWERING_WITH_SUMMARY_IMAGE:
            logger.info(f"Summarizing images for page: {confluence_object['title']}")
            # get images from page
            page_images = asyncio.run(
                self._summarize_page_images(
                    confluence_object, self.confluence_client, USER_PROMPT
                )
            )
            # add tag to flag summaries (needed to switch between base and multimodal danswer)
            doc_metadata["is_image_summary"] = "True"

            # if page contains any images: add caption of each image to document
            if page_images:
                for image in page_images:
                    if MULTIMODAL_ANSWERING_WITH_RAW_IMAGE:
                        doc_metadata["image"] = image.base64_encoded

                    image_docs.append(
                        Document(
                            id=image.url,
                            sections=[
                                Section(link=object_url, text=image.summary or "")
                            ],
                            source=DocumentSource.CONFLUENCE,
                            semantic_identifier=image.title,
                            doc_updated_at=last_modified,
                            primary_owners=(
                                [BasicExpertInfo(email=author_email)]
                                if author_email
                                else None
                            ),
                            metadata=doc_metadata,
                        )
                    )
                logger.debug(
                    f"Added {len(page_images)} image documents for page: {confluence_object['title']}"
                )

        return doc, image_docs

    def _fetch_document_batches(self) -> GenerateDocumentsOutput:
        if self.confluence_client is None:
            raise ConnectorMissingCredentialError("Confluence")

        doc_batch: list[Document] = []
        confluence_page_ids: list[str] = []

        page_query = self.cql_page_query + self.cql_label_filter + self.cql_time_filter
        # Fetch pages as Documents
        for page_batch in self.confluence_client.paginated_cql_page_retrieval(
            cql=page_query,
            expand=",".join(_PAGE_EXPANSION_FIELDS),
            limit=self.batch_size,
        ):
            for page in page_batch:
                confluence_page_ids.append(page["id"])
                doc, image_docs = self._convert_object_to_document(page)

                if doc is not None:
                    doc_batch.append(doc)
                if image_docs:
                    doc_batch.extend(image_docs)

                if len(doc_batch) >= self.batch_size:
                    yield doc_batch
                    doc_batch = []

        # Fetch attachments as Documents
        for confluence_page_id in confluence_page_ids:
            attachment_cql = f"type=attachment and container='{confluence_page_id}'"
            attachment_cql += self.cql_label_filter
            # TODO: maybe should add time filter as well?
            for attachments in self.confluence_client.paginated_cql_page_retrieval(
                cql=attachment_cql,
                expand=",".join(_ATTACHMENT_EXPANSION_FIELDS),
            ):
                for attachment in attachments:
                    doc, image_docs = self._convert_object_to_document(attachment)
                    if doc is not None:
                        doc_batch.append(doc)
                    if image_docs:
                        doc_batch.extend(image_docs)

                    if len(doc_batch) >= self.batch_size:
                        yield doc_batch
                        doc_batch = []

        if doc_batch:
            yield doc_batch

    def load_from_state(self) -> GenerateDocumentsOutput:
        return self._fetch_document_batches()

    def poll_source(self, start: float, end: float) -> GenerateDocumentsOutput:
        # Add time filters
        formatted_start_time = datetime.fromtimestamp(start, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
        formatted_end_time = datetime.fromtimestamp(end, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
        self.cql_time_filter = f" and lastmodified >= '{formatted_start_time}'"
        self.cql_time_filter += f" and lastmodified <= '{formatted_end_time}'"
        return self._fetch_document_batches()

    def retrieve_all_slim_documents(
        self,
        start: SecondsSinceUnixEpoch | None = None,
        end: SecondsSinceUnixEpoch | None = None,
    ) -> GenerateSlimDocumentOutput:
        if self.confluence_client is None:
            raise ConnectorMissingCredentialError("Confluence")

        doc_metadata_list: list[SlimDocument] = []

        restrictions_expand = ",".join(_RESTRICTIONS_EXPANSION_FIELDS)

        page_query = self.cql_page_query + self.cql_label_filter
        for pages in self.confluence_client.cql_paginate_all_expansions(
            cql=page_query,
            expand=restrictions_expand,
        ):
            for page in pages:
                # If the page has restrictions, add them to the perm_sync_data
                # These will be used by doc_sync.py to sync permissions
                perm_sync_data = {
                    "restrictions": page.get("restrictions", {}),
                    "space_key": page.get("space", {}).get("key"),
                }

                doc_metadata_list.append(
                    SlimDocument(
                        id=build_confluence_document_id(
                            self.wiki_base,
                            page["_links"]["webui"],
                            self.is_cloud,
                        ),
                        perm_sync_data=perm_sync_data,
                    )
                )
                attachment_cql = f"type=attachment and container='{page['id']}'"
                attachment_cql += self.cql_label_filter
                for attachments in self.confluence_client.cql_paginate_all_expansions(
                    cql=attachment_cql,
                    expand=restrictions_expand,
                ):
                    for attachment in attachments:
                        doc_metadata_list.append(
                            SlimDocument(
                                id=build_confluence_document_id(
                                    self.wiki_base,
                                    attachment["_links"]["webui"],
                                    self.is_cloud,
                                ),
                                perm_sync_data=perm_sync_data,
                            )
                        )
                yield doc_metadata_list
                doc_metadata_list = []

    @classmethod
    def _attachment_to_download_link(
        cls, confluence_client: Confluence, attachment: dict[str, Any]
    ) -> str:
        return confluence_client.url + attachment["_links"]["download"]

    @classmethod
    async def _summarize_page_images(
        cls, page: Dict[str, Any], confluence_client: Confluence, USER_PROMPT: str
    ) -> List[ImageSummarization]:
        """Create LLM summaries of all embedded (used) image attachments on the given page"""

        page_id = page["id"]
        confluence_xml = page["body"]["storage"]["value"]
        attachments = cls._get_embedded_image_attachments(
            confluence_client, confluence_xml, page_id
        )

        # TODO: Handling of image not present in attachments...(?)
        # image_urls_test = re.findall(r'ac:src="([^"]+)"', confluence_xml)
        # logger.warning(f"image_urls_test: {image_urls_test}")

        async def summarize_attachment(attachment, USER_PROMPT):
            title = attachment["title"]
            download_link = ConfluenceConnector._attachment_to_download_link(
                confluence_client, attachment
            )
            logger.info(f"download_link = {download_link}")

            try:
                # get image from url
                image_data = confluence_client.get(
                    download_link, absolute=True, not_json_response=True
                )
            except requests.exceptions.RequestException as e:
                logger.error(
                    "Failed to fetch image for summarization. url=%s",
                    download_link,
                    exc_info=e,
                )
                return None

            # get image summary
            # format user prompt: add page title and XML content of page to enable a better summarization of the llm
            USER_PROMPT = USER_PROMPT.format(title=title, page_title=page["title"])
            image_context = USER_PROMPT + confluence_xml
            summary = summarize_image(image_data, image_context, SYSTEM_PROMPT)

            base64_image = base64.b64encode(image_data).decode("utf-8")

            return ImageSummarization(
                url=download_link,
                title=title,
                base64_encoded=base64_image,
                media_type=attachment["metadata"]["mediaType"],
                summary=summary,
            )

        results = await asyncio.gather(
            *[
                summarize_attachment(attachment, USER_PROMPT)
                for attachment in attachments
            ]
        )

        return [result for result in results if result is not None]

    @classmethod
    def _get_embedded_image_attachments(
        cls, confluence_client: Confluence, confluence_xml: str, page_id: str
    ) -> List[Dict[str, Any]]:
        page_attachments = cls._get_page_attachments(confluence_client, page_id)

        relevant_tags = SoupStrainer(["ac:image", "ac:structured-macro"])
        soup = bs4.BeautifulSoup(
            confluence_xml, "html.parser", parse_only=relevant_tags
        )

        image_attachment_tags = soup.find_all(
            lambda tag: tag.name == "ri:attachment"
            and tag.parent is not None
            and tag.parent.name == "ac:image"
        )
        image_attachments = [
            att
            for att in page_attachments
            if att["title"] in [tag["ri:filename"] for tag in image_attachment_tags]
            and att["metadata"]["mediaType"].startswith("image/")
        ]

        gliffy_macro_tags = soup.find_all(
            "ac:structured-macro", attrs={"ac:name": "gliffy"}
        )
        gliffy_attachments = [
            att
            for att in page_attachments
            if att["id"]
            in [
                tag.find(attrs={"ac:name": "imageAttachmentId"}).string
                for tag in gliffy_macro_tags
                if tag.find(attrs={"ac:name": "imageAttachmentId"}) is not None
            ]
        ]

        return [*image_attachments, *gliffy_attachments]

    @classmethod
    def _get_page_attachments(
        cls, confluence_client: Confluence, page_id: str
    ) -> List[Dict[str, Any]]:
        get_attachments_from_content = handle_confluence_rate_limit(
            confluence_client.get_attachments_from_content
        )
        expand = "history.lastUpdated,metadata.labels"
        attachments_container = get_attachments_from_content(
            page_id, start=0, limit=500, expand=expand
        )
        attachments = attachments_container["results"]
        return attachments
