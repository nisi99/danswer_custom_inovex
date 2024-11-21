import base64
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from litellm import BadRequestError
from openai import RateLimitError
from PIL import Image

from danswer.file_processing.image_summarization import _encode_image
from danswer.file_processing.image_summarization import _resize_image_if_needed
from danswer.file_processing.image_summarization import summarize_image


# Mock LLM class for testing
class MockLLM:
    def invoke(self, messages):
        # Simulate a response object with a 'content' attribute
        class Response:
            def __init__(self, content):
                self.content = content

        # Simulate successful invocation
        return Response("This is a summary of the image.")


# Helper function to create a dummy image
def create_image(size: tuple, color: str, format: str) -> bytes:
    img = Image.new("RGB", size, color)
    output = BytesIO()
    img.save(output, format=format)
    return output.getvalue()


def test_encode_different_image_formats():
    """Tests the base64 encoding of different image formats."""
    formats = ["JPEG", "PNG", "GIF", "EPS"]
    for format in formats:
        image_data = create_image((100, 100), "blue", format)

        expected_output = "data:image/jpeg;base64," + base64.b64encode(
            image_data
        ).decode("utf-8")

        result = _encode_image(image_data)

        assert result == expected_output


def test_resize_image_above_max_size():
    """Test that an image above the max size is resized."""
    image_data = create_image((2000, 2000), "red", "JPEG")  # Large image
    result = _resize_image_if_needed(image_data, max_size_mb=1)

    # Check if the resized image is below the max size
    assert len(result) < (1 * 1024 * 1024)  # Size should be less than 1 MB


def test_summarization_of_images():
    """Test that summarize_image returns a valid summary."""
    encoded_image = "data:image/jpeg;base64,idFuHHIwEEOHVAA..."
    query = "What is in this image?"
    system_prompt = "You are a helpful assistant."
    llm = MockLLM()

    result = summarize_image(
        encoded_image=encoded_image, query=query, system_prompt=system_prompt, llm=llm
    )
    assert result == "This is a summary of the image."


# Mock response for RateLimitError
class MockResponse:
    def __init__(self):
        self.request = "mock_request"  # Simulate the request attribute
        self.status_code = 429
        self.headers = {"x-request-id": "mock_request_id"}


@pytest.mark.parametrize(
    "exception, expected_output",
    [
        (
            BadRequestError(
                "Content policy violation",
                model="model_name",
                llm_provider="provider_name",
            ),
            "Summarization failed with error: litellm.BadRequestError: Content policy violation.",
        ),
        (
            RateLimitError(
                "Retry limit exceeded", response=MockResponse(), body="body"
            ),
            "Summarization failed with error: Retry limit exceeded.",
        ),
    ],
)
def test_summarize_image_raises_value_error_on_failure(
    mocker, exception, expected_output
):
    llm = MockLLM()

    global CONTINUE_ON_CONNECTOR_FAILURE
    CONTINUE_ON_CONNECTOR_FAILURE = False

    # Set the LLM invoke method to raise the specified exception
    llm.invoke = MagicMock(side_effect=exception)

    # Use pytest.raises to assert that the exception is raised
    with pytest.raises(ValueError) as excinfo:
        summarize_image("encoded_image_string", llm, "test query", "system prompt")

    # Assert that the exception message matches the expected message
    assert expected_output == str(excinfo.value)


# def test_summarize_image_return_none_on_failure(mocker, exception):
#     llm = MockLLM()

#     global CONTINUE_ON_CONNECTOR_FAILURE
#     CONTINUE_ON_CONNECTOR_FAILURE = True

#     # Mock the invoke method to raise a BadRequestError
#     llm.invoke = MagicMock(side_effect=exception)

#     # Call the summarize_image function
#     result = summarize_image("encoded_image_string", llm, "test query", "system prompt")
#     print(result)
#     # Assert that the result is None
#     assert result is None
