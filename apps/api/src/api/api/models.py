from pydantic import BaseModel, Field
from typing import Optional


class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")

class RAGUsedContext(BaseModel):
    image_url: str = Field(description="URL of the image of the item used to answer the question")
    price: Optional[float] = Field(description="Price of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The answer to the query")
    used_context: list[RAGUsedContext] = Field(description="List of items used to answer the question")


# class RAGRequest(BaseModel):
#     query: str


# class RAGResponse(BaseModel):
#     answer: str = Field(description="Answer to the question.")
#     used_context: list[RAGUsedContext] = Field(description="List of items used to answer the question")
