from fastapi import Request, APIRouter
from .models import RAGRequest, RAGResponse

from api.agents.retrieval_generation import rag_pipeline

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()

@rag_router.post("/")
def rag(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:

    answer = rag_pipeline(payload.query)

    return RAGResponse(
        request_id=request.state.request_id,
        answer=answer,
    )


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])