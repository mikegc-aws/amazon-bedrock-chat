import boto3
import json
from botocore.exceptions import ClientError
from pydantic import BaseModel
from typing import Literal, Dict, Optional

from jinja2 import Template

class MistralLLM(BaseModel):

    modelId: Literal[
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral.mistral-large-2402-v1:0"
    ] = "mistral.mistral-large-2402-v1:0"
    region_name: str = "us-west-2"

    content_type: str = "application/json"
    accept_type: str = "application/json"

    max_tokens: int = 1000
    temperature: float = 0.5
    top_p: float = 0.9
    top_k: int = 50
    bos_token: str = "<s>"
    eos_token: str = "</s>"

    prompt_template: str = "{{ bos_token }}{% set first_user_message_handled = false %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{% if not first_user_message_handled %}{{ '[INST] ' + system_prompt + ' ' + message['content'][0]['text'] + ' [/INST]' }}{% set first_user_message_handled = true %}{% else %}{{ '[INST] ' + message['content'][0]['text'] + ' [/INST]' }}{% endif %}{% elif message['role'] == 'assistant' %}{{ message['content'][0]['text'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize the AWS Bedrock Runtime client with the specified AWS region.
        self._bedrock_runtime = boto3.client("bedrock-runtime", region_name=self.region_name)

    @property
    def bedrock_runtime(self):
        """Provides access to the Bedrock Runtime client."""
        return self._bedrock_runtime

    def _prepare_kwargs(self, prompt: str, system_prompt: Optional[str] = None) -> Dict:
        """
        Prepare common keyword arguments for API calls.

        Parameters:
            prompt (Dict): The input prompt for the model.
            system_prompt (Optional[str]): An optional system-level prompt to prepend.

        Returns:
            Dict: A dictionary of keyword arguments ready for the API call.
        """

        return {
            "modelId": self.modelId,
            "contentType": self.content_type,
            "accept": self.accept_type,
            "body": json.dumps({
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "prompt": prompt
            })
        }

    def _format_prompt_as_string(self, prompt: Dict, system_prompt: str = ""):
        """
        Format the prompt as a string.

        Parameters:
            prompt (Dict): The input prompt for the model.

        Returns:
            str: The formatted prompt as a string.
        """
        # Load the template string into a Jinja object. 
        template = Template(self.prompt_template)

        data = {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "system_prompt": system_prompt,
            "messages" : prompt['messages']
        }

        # Combine the data with the template.
        rendered_template = template.render(data)

        # Take a look at what we got!
        return rendered_template
    

    def generate(self, prompt: Dict, system_prompt: str = None):
        """
        Generate a response synchronously based on the given prompt.

        Parameters:
            prompt (Dict): The input prompt for the model.
            system_prompt (str, optional): An optional system-level prompt to prepend.

        Returns:
            Dict: A dictionary with the 'role' and 'content' of the generated response.
        """
        prompt_string = self._format_prompt_as_string(prompt, system_prompt)
        kwargs = self._prepare_kwargs(prompt_string)
        response = self.bedrock_runtime.invoke_model(**kwargs)
        body = json.loads(response.get("body").read())

        return {"role": "assistant", "content": [{"type": "text", "text": body['outputs'][0]['text']}]}

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
        chunk = ""
        try:
            for event in response.get("body"):
                event_data = json.loads(event["chunk"]["bytes"])

                outputs = event_data.get("outputs", {})
                chunk = outputs[0].get("text", "")

                message_text += chunk
                message = {"role": "assistant", "content": [{"type": "text", "text": message_text}]}
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
            prompt_string = self._format_prompt_as_string(prompt, system_prompt)
            kwargs = self._prepare_kwargs(prompt_string)

            # print(json.dumps(kwargs, indent=2))

            response = self.bedrock_runtime.invoke_model_with_response_stream(**kwargs)
            yield from self.stream_response_chunks(response)
        except ClientError as e:
            print(f"An error occurred: {e}")
            raise