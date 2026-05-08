from pydantic import BaseModel, Field
from typing import Optional, Union


class RAGRequest(BaseModel):
    query: str
    thread_id: str

class RAGUsedContext(BaseModel):
    image_url: str = Field(description="URL of the image of the item used to answer the question")
    price: Optional[float] = Field(description="Price of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class RAGResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    used_context: list[RAGUsedContext] = Field(description="List of items used to answer the question")
    trace_id: str = Field(description="Trace ID of the question")


class FeedbackRequest(BaseModel):
    trace_id: str
    feedback_score: Union[int, None] = Field(description="Feedback score")
    feedback_text: str = Field(description="Feedback text")
    feedback_source_type: str = Field(description="Feedback source type")

class FeedbackResponse(BaseModel):
    message: str = Field(description="Message indicating that the feedback has been submitted")
