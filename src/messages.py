import json
from pydantic import BaseModel, ValidationError, validator
from typing import Optional, List, Union, Literal

class MessageContentSource(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: Literal["image/jpeg", "image/png"] = "image/png"
    data: str

class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    source: MessageContentSource

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

Content = Union[ImageContent, TextContent]

class Message(BaseModel):
    role: Optional[Literal["user", "assistant"]] = "user"
    content: List[Content]

    @validator('content', pre=True, each_item=True)
    def default_to_textcontent(cls, v):
        if isinstance(v, str):
            # If the item is a string, wrap it in TextContent
            return TextContent(type='text', text=v)
        return v

class Messages(BaseModel):
    messages: List[Message] = []

    def add(self, payload):

        if isinstance(payload, str):
            # Directly handle string content by wrapping it into the expected structure
            payload = {"role" : "user", "content" :[{"type": "text", "text": payload}]}
        try:
            new_message = Message(**payload)
            self.messages.append(new_message)
        except ValidationError as e:
            print("Validation error:", e)

    def update_stream(self, streaming_message):
        try:
            streaming_message_role = streaming_message.get("role", "assistant")
            if streaming_message_role == self.messages[-1].role:
                self.messages[-1] = Message(**streaming_message)
            else:
                self.messages.append(Message(**streaming_message))
        except ValidationError as e:
            print("Validation error:", e)