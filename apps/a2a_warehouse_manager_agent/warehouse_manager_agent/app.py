import logging
import os

import uvicorn
from starlette.applications import Starlette

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory import InMemoryMemoryService

from agent import WarehouseManagerAgent
from agent_executor import WarehouseManagerAgentExecutor
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = "localhost"
PORT = 10001


def main():

    capabilities = AgentCapabilities(streaming=True)

    skill_availability = AgentSkill(
        id="ABC",
        name="Check Availability",
        description="Check the availability of items in the warehouses",
        tags=["warehouse", "availability"],
        examples=["what is the availability of the item 123?"]
    ) 
    skill_reservation = AgentSkill(
        id="DEF",
        name="Reserve Items",
        description="Reserve items in the warehouses",
        tags=["warehouse", "reservation"],
        examples=["reserve 10 items of the item 123 in the Berlin warehouse."]
    )

    agent_card = AgentCard(
        name="warehouse_manager_agent",
        description="An agent that can check the availability of items in the warehouses and reserve them.",
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=f"http://{HOST}:{PORT}/",
            )
        ],
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=capabilities,
        skills=[skill_availability, skill_reservation]
    )

    adk_agent = WarehouseManagerAgent().get_agent()
    runner = Runner(
        agent=adk_agent,
        app_name=agent_card.name,
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        memory_service=InMemoryMemoryService()
    )
    agent_executor = WarehouseManagerAgentExecutor(runner)

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(request_handler, rpc_url="/"))

    app = Starlette(routes=routes)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()