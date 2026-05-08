from fastapi import APIRouter, Request
from api.api.models import RAGRequest, RAGResponse, RAGUsedContext, FeedbackRequest, FeedbackResponse
from api.agents.graph import agent_wrapper
from api.api.processors.submit_feedback import submit_feedback

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()
feedback_router = APIRouter()

@rag_router.post("/")
def chat(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:

    result = agent_wrapper(payload.query, payload.thread_id)

    return RAGResponse(
        answer=result["answer"], 
        used_context=[RAGUsedContext(**item) for item in result["used_context"]],
        trace_id=result["trace_id"]
    )

@feedback_router.post("/")
def send_feedback(
    request: Request,
    payload: FeedbackRequest
) -> FeedbackResponse:

    submit_feedback(payload.trace_id, payload.feedback_score, payload.feedback_text, payload.feedback_source_type)

    return FeedbackResponse(
        message="Success"
    )

api_router = APIRouter()
api_router.include_router(rag_router, prefix="/agent", tags=["rag"])
api_router.include_router(feedback_router, prefix="/submit_feedback", tags=["feedback"])