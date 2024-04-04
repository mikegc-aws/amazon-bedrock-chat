import boto3
import json
from botocore.exceptions import ClientError
from pydantic import BaseModel
from typing import Literal, Dict, Optional

class ClaudeLLM(BaseModel):
    """
    A class representing Claude Language Models, facilitating synchronous and
    asynchronous interactions with various versions of the Claude LLM via the 
    AWS Bedrock Runtime.

    Attributes:
        modelId (Literal): Identifier for the specific Claude model version.
        region_name (str): AWS region where the Bedrock Runtime is available.
        content_type (str): Content type for the request, typically "application/json".
        accept_type (str): Expected content type of the response, typically "application/json".
        anthropic_version (str): Version of the Anthropic API being used.
        max_tokens (int): Maximum number of tokens to generate in the response.

    Methods:
        generate(prompt, system_prompt=None): Generate response synchronously.
        generate_stream(prompt, system_prompt=None): Stream response asynchronously, yielding chunks.
    """
    modelId: Literal[
        "anthropic.claude-instant-v1",
        "anthropic.claude-v2:1",
        "anthropic.claude-v2",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ] = "anthropic.claude-3-sonnet-20240229-v1:0"
    region_name: str = "us-west-2"
    content_type: str = "application/json"
    accept_type: str = "application/json"
    anthropic_version: str = "bedrock-2023-05-31"
    max_tokens: int = 1000

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize the AWS Bedrock Runtime client with the specified AWS region.
        self._bedrock_runtime = boto3.client("bedrock-runtime", region_name=self.region_name)

    @property
    def bedrock_runtime(self):
        """Provides access to the Bedrock Runtime client."""
        return self._bedrock_runtime

    def _prepare_kwargs(self, prompt: Dict, system_prompt: Optional[str] = None) -> Dict:
        """
        Prepare common keyword arguments for API calls.

        Parameters:
            prompt (Dict): The input prompt for the model.
            system_prompt (Optional[str]): An optional system-level prompt to prepend.

        Returns:
            Dict: A dictionary of keyword arguments ready for the API call.
        """
        if system_prompt:
            prompt['system'] = system_prompt

        return {
            "modelId": self.modelId,
            "contentType": self.content_type,
            "accept": self.accept_type,
            "body": json.dumps({
                "anthropic_version": self.anthropic_version,
                "max_tokens": self.max_tokens,
                **prompt
            })
        }

    def generate(self, prompt: Dict, system_prompt: str = None):
        """
        Generate a response synchronously based on the given prompt.

        Parameters:
            prompt (Dict): The input prompt for the model.
            system_prompt (str, optional): An optional system-level prompt to prepend.

        Returns:
            Dict: A dictionary with the 'role' and 'content' of the generated response.
        """
        kwargs = self._prepare_kwargs(prompt, system_prompt)
        response = self.bedrock_runtime.invoke_model(**kwargs)
        body = json.loads(response.get("body").read())
        return {"role": body['role'], "content": body['content']}

    def stream_response_chunks(self, response):
        """
        Stream chunks from the response body and process each chunk.

        Parameters:
            response: The response object from an asynchronous invocation.

        Yields:
            Tuple[str, Dict]: Each yield provides a chunk of text and the cumulative message content.
        """
        message = {}
        message_text = ""
        message_start_info = {}
        chunk = ""
        try:
            for event in response.get("body"):
                event_data = json.loads(event["chunk"]["bytes"])
                event_type = event_data.get("type")
                if event_type == "message_start":
                    message_start_info = event_data.get("message")
                elif event_type in ["content_block_delta", "content_block_start"]:
                    chunk = event_data.get("delta", {}).get("text", "")
                    message_text += chunk
                    message = {"role": message_start_info.get("role", ""), "content": [{"type": "text", "text": message_text}]}
                    yield chunk, message
        except json.JSONDecodeError as e:
            print("\nError decoding JSON from response chunk:", e)
            raise

    def generate_stream(self, prompt: Dict, system_prompt: str = None):
        """
        Invoke the model with the given prompt and stream the response asynchronously.

        Parameters:
            prompt (Dict): The input prompt for the model.
            system_prompt (str, optional): An optional system-level prompt to prepend.

        Yields:
            The streaming response, chunk by chunk.
        """
        try:
            kwargs = self._prepare_kwargs(prompt, system_prompt)
            response = self.bedrock_runtime.invoke_model_with_response_stream(**kwargs)
            yield from self.stream_response_chunks(response)
        except ClientError as e:
            print(f"An error occurred: {e}")
            raise
