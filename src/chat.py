import json
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Literal
from .messages import Messages
from .llm import ClaudeLLM
    

class Chat(BaseModel):
    messages: Messages = Field(default_factory=Messages)
    system_prompt: Optional[str] = None
    llm: ClaudeLLM = Field(default_factory=ClaudeLLM)

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self, prompt):

        self.messages.add(prompt)
        
        response = self.llm.generate(
            self.messages.dict(), 
            system_prompt=self.system_prompt
        )

        self.messages.add(response)

        if len(response.get('content')) > 0:
            if "text" in response.get('content')[0]:
                return response.get('content')[0]['text']
        else: 
            return "-"
        
    def stream(self, prompt):
        self.messages.add(prompt)
        
        yield from self.llm.generate_stream(
            self.messages.dict(), 
            system_prompt=self.system_prompt
        )

    def generate_stream(self, prompt):
        for chunk, message in (x for x in self.stream(prompt)):
            yield chunk
            self.messages.update_stream(message)

    def reset(self):
        self.messages = Messages()