from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from api.api.models import RAGRequest, HitlRequest, RAGUsedContext, FeedbackRequest, FeedbackResponse
from api.agents.graph import agent_stream_wrapper
from api.api.processors.submit_feedback import submit_feedback

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()
feedback_router = APIRouter()
hitl_router = APIRouter()

@rag_router.post("/")
def chat(
    request: Request,
    payload: RAGRequest
) -> StreamingResponse:

    return StreamingResponse(
        agent_stream_wrapper(payload.query, payload.thread_id, "initialise"),
        media_type="text/event-stream"
    )

@hitl_router.post("/")
def hitl(
    request: Request,
    payload: HitlRequest
) -> StreamingResponse:

    return StreamingResponse(
        agent_stream_wrapper(payload.approved, payload.thread_id, "hitl"),
        media_type="text/event-stream"
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
api_router.include_router(hitl_router, prefix="/send_hitl_response", tags=["send_hitl_response"])