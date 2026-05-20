import logging
from collections.abc import AsyncGenerator

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TaskState, UnsupportedOperationError
from google.adk import Runner
from google.adk.events import Event
from google.genai import types

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class WarehouseManagerAgentExecutor(AgentExecutor):

    def __init__(self, runner: Runner):
        self.runner = runner
        self._running_sessions = {}

    def _run_agent(
        self, session_id, new_message: types.Content
    ) -> AsyncGenerator[Event, None]:
        return self.runner.run_async(
            session_id=session_id,
            user_id="warehouse_manager_agent",
            new_message=new_message
        )

    async def _process_request(
        self,
        new_message: types.Content,
        session_id: str,
        task_updater: TaskUpdater,
    ) -> None:
        session_obj = await self._upsert_session(session_id)
        session_id = session_obj.id
    
        async for event in self._run_agent(session_id, new_message):
            if event.is_final_response():
                parts = convert_genai_parts_to_a2a(
                    event.content.parts if event.content and event.content.parts else []
                )
                logger.debug("Yielding final response: %s", parts)
                await task_updater.add_artifact(parts)
                await task_updater.complete()
                break
            if not event.get_function_calls():
                logger.debug("Yielding update response")
                task_updater.update_status(
                    TaskState.TASK_STATE_WORKING,
                    message=task_updater.new_agent_message(
                        convert_genai_parts_to_a2a(
                            event.content.parts
                            if event.content and event.content.parts
                            else []
                        ),
                    ),
                )
            else:
                logger.debug("Skipping event")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")
        
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            updater.submit()
        updater.start_work()
        await self._process_request(
            types.UserContent(
                parts=convert_a2a_parts_to_genai(context.message.parts),
            ),
            context.context_id,
            updater,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise SystemError(error=UnsupportedOperationError())
    
    async def _upsert_session(self, session_id: str):
        session = await self.runner.session_service.get_session(
            app_name=self.runner.app_name,
            user_id="warehouse_manager_agent",
            session_id=session_id
        )
        if session is None:
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name,
                user_id="warehouse_manager_agent",
                session_id=session_id
            )
        if session is None:
            raise RuntimeError(f"Failed to get or create session: {session_id}")
        return session






def convert_a2a_parts_to_genai(parts: list[Part]) -> list[types.Part]:
    """Convert a list of A2A Part types into a list of Google Gen AI Part types."""
    return [convert_a2a_part_to_genai(part) for part in parts]


# def convert_a2a_part_to_genai(part: Part) -> types.Part:
#     """Convert a single A2A Part type into a Google Gen AI Part type."""
#     root = part.root
#     if isinstance(root, TextPart):
#         return types.Part(text=root.text)
#     if isinstance(root, FilePart):
#         if isinstance(root.file, FileWithUri):
#             return types.Part(
#                 file_data=types.FileData(
#                     file_uri=root.filesuri, mime_type=root.file.mimeType
#                 )
#             )
#         if isinstance(root.file, FileWithBytes):
#             return types.Part(
#                 inline_data=types.Blob(
#                     data=root.file.bytes.encode("utf-8"),
#                     mime_type=root.file.mimeType or "application/octet-stream",
#                 ) 
#             )
#         raise ValueError(f"Unsupported file type: {type(root.file)}")
#     raise ValueError(f"Unsupported part type: {type(part)}")


def convert_a2a_part_to_genai(part: Part) -> types.Part:
    """Convert a single A2A Part type into a Google Gen AI Part type."""
    if part.HasField("text"):
        return types.Part(text=part.text)
    if part.HasField("url"):
        return types.Part(
            file_data=types.FileData(
                file_uri=part.url,
                mime_type=part.media_type or "application/octet-stream",
            )
        )
    if part.HasField("raw"):
        return types.Part(
            inline_data=types.Blob(
                data=part.raw,
                mime_type=part.media_type or "application/octet-stream",
            )
        )
    raise ValueError(f"Unsupported part type: {part}")


def convert_genai_parts_to_a2a(parts: list[types.Part]) -> list[Part]:
    """Convert a list of Google Gen AI Part types into a list of A2A Part types."""
    return [
        convert_genai_part_to_a2a(part)
        for part in parts
        if (part.text or part.file_data or part.inline_data)
    ]


# def convert_genai_part_to_a2a(part: types.Part) -> Part:
#     """Convert a single Google Gen AI Part type into an A2A Part type."""
#     if part.text:
#         return Part(root=TextPart(text=part.text))
#     if part.file_data:
#         if not part.file_data.file_uri:
#             raise ValueError("File URI is missing")
#         return Part(
#             root=FilePart(
#                 file=FileWithUri(
#                     uri=part.file_data.file_uri,
#                     mimeType=part.file_data.mime_type,
#                 )
#             )
#         )
#     if part.inline_data:
#         if not part.inline_data.data:
#             raise ValueError("Inline data is missing")
#         return Part(
#             root=FilePart(
#                 file=FileWithBytes(
#                     bytes=part.inline_data.data.decode("utf-8"),
#                     mimeType=part.inline_data.mime_type,
#                 )
#             )
#         )
#     raise ValueError("Unsupported part type: {part}")

def convert_genai_part_to_a2a(part: types.Part) -> Part:
    """Convert a single Google Gen AI Part type into an A2A Part type."""
    if part.text:
        return Part(text=part.text)
    if part.file_data:
        if not part.file_data.file_uri:
            raise ValueError("File URI is missing")
        return Part(
            url=part.file_data.file_uri,
            media_type=part.file_data.mime_type,
        )
    if part.inline_data:
        if not part.inline_data.data:
            raise ValueError("Inline data is missing")
        return Part(
            raw=part.inline_data.data,  # already bytes, no decode needed
            media_type=part.inline_data.mime_type or "application/octet-stream",
        )
    raise ValueError(f"Unsupported part type: {part}")