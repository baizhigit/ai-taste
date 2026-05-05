from fastapi import APIRouter, Request
from api.api.models import RAGRequest, RAGResponse, RAGUsedContext
from api.agents.graph import agent_wrapper

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()

@rag_router.post("/")
def chat(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:

    result = agent_wrapper(payload.query)

    return RAGResponse(
        answer=result["answer"], 
        used_context=[RAGUsedContext(**item) for item in result["used_context"]]
    )

api_router = APIRouter()
api_router.include_router(rag_router, prefix="/agent", tags=["rag"])