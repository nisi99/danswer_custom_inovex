from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from danswer.connectors.confluence.connector import ConfluenceConnector

CONFLUENCE_IMAGE_SUMMARIZATION_MULTIMODAL_ANSWERING = True


# Mocking the LLM and its methods
class MockLLM:
    def __init__(self, vision_support):
        self._vision_support = vision_support

    def vision_support(self):
        return self._vision_support


# Mocking the get_default_llms function
def mock_get_default_llms(vision_support):
    return MockLLM(vision_support), None


@pytest.mark.parametrize(
    "vision_support, expected_exception",
    [
        (True, None),  # Successful case
        (False, ValueError),  # Not multimodal
        (None, ValueError),  # vision_support not defined
    ],
)
def test_validate_llm_configuration_vision_support(vision_support, expected_exception):
    with patch("danswer.llm.factory.get_default_llms"):
        llm, _ = mock_get_default_llms(vision_support)

        # Create an instance of LLMChecker
        connector = ConfluenceConnector(wiki_base="https://example.com", is_cloud=True)

        if expected_exception:
            with pytest.raises(expected_exception) as excinfo:
                connector.validate_llm(llm)
                assert "Your default LLM seems to be not multimodal." in str(
                    excinfo.value
                )
        else:
            connector.validate_llm(llm)


@pytest.fixture
def mock_confluence_client():
    """Fixture to create a mock Confluence client."""
    return MagicMock()


def test_get_page_attachments(mock_confluence_client):
    page_id = "12345"
    expected_attachments = [
        {"id": "attach1", "title": "Attachment 1"},
        {"id": "attach2", "title": "Attachment 2"},
    ]

    # Mock the return value of get_attachments_from_content
    mock_confluence_client.get_attachments_from_content.return_value = {
        "results": expected_attachments
    }

    attachments = ConfluenceConnector._get_page_attachments(
        mock_confluence_client, page_id
    )

    assert attachments == expected_attachments


def test_no_attachments(mock_confluence_client):
    # Arrange
    page_id = "12345"
    confluence_xml = "<document></document>"

    ConfluenceConnector._get_page_attachments = MagicMock(return_value=[])

    # Act
    result = ConfluenceConnector._get_embedded_image_attachments(
        mock_confluence_client, confluence_xml, page_id
    )

    # Assert
    assert result == []


def test_get_embedded_image_attachments(mock_confluence_client):
    page_id = "12345"
    confluence_xml = """
    <document>
        <ac:image>
            <ri:attachment ri:filename="image1.png"/>
        </ac:image>
        <ac:structured-macro ac:name="gliffy">
            <ac:parameter ac:name="imageAttachmentId">attach1</ac:parameter>
        </ac:structured-macro>
    </document>
    """

    expected_attachments = [
        {
            "id": "attach1",
            "title": "image1.png",
            "metadata": {"mediaType": "image/png"},
        },
        {
            "id": "attach2",
            "title": "document.pdf",
            "metadata": {"mediaType": "application/pdf"},
        },
    ]

    # Mock the return value of _get_page_attachments
    ConfluenceConnector._get_page_attachments = MagicMock(
        return_value=expected_attachments
    )

    # Act: Call the synchronous method directly
    result = ConfluenceConnector._get_embedded_image_attachments(
        mock_confluence_client, confluence_xml, page_id
    )
    print(f"result = {result}")
    # Assert
    expected_result = [
        {
            "id": "attach1",
            "title": "image1.png",
            "metadata": {"mediaType": "image/png"},
        },
    ]

    assert result == expected_result
