from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, Annotated, List
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Send, Command, interrupt
from api.agents.tools import get_formatted_items_context, get_formatted_reviews_context, add_to_shopping_cart, remove_from_cart, get_shopping_cart, check_warehouse_availability, reserve_warehouse_items
from api.agents.agents import ToolCall, RAGUsedContext, Delegation, product_qa_agent, shopping_cart_agent, warehouse_manager_agent, coordinator_agent
from api.agents.utils.utils import get_tool_descriptions
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
from operator import add
from langgraph.checkpoint.postgres import PostgresSaver
from langsmith import traceable
import json


class AgentPreporties(BaseModel):
    iteration: int = 0
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []
    final_answer: bool = False

class CoordinatorAgentPreporties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    plan: List[Delegation] = []
    next_agent: str = ""

class State(BaseModel):
    messages: Annotated[List[Any], add_messages] = []
    user_intent: str = ""
    product_qa_agent: AgentPreporties = Field(default_factory=AgentPreporties)
    shopping_cart_agent: AgentPreporties = Field(default_factory=AgentPreporties)
    warehouse_manager_agent: AgentPreporties = Field(default_factory=AgentPreporties)
    coordinator_agent: CoordinatorAgentPreporties = Field(default_factory=CoordinatorAgentPreporties)
    answer: str = ""
    references: Annotated[List[RAGUsedContext], add] = []
    user_id: str = ""
    cart_id: str = ""
    trace_id: str = ""



### HITL Nodes

@traceable(
    name="hitl_add_to_cart")
def hitl_add_to_cart(state) -> Command[Literal["shopping_cart_agent_tool_node", END]]:

    for tool_call in state.shopping_cart_agent.tool_calls:
        if tool_call.name == "add_to_shopping_cart":
            items_to_add = tool_call.arguments["items"]
            break

    human_input = interrupt({
        "items_to_add": items_to_add
    })

    if human_input.get("confirmed"):
        return Command(
            update = {},
            goto = "shopping_cart_agent_tool_node"
        )
    else:

        last_msg = state.messages[-1]
        sanitazed = AIMessage(
            content=last_msg.content,
            id=last_msg.id
        )

        return Command(
            update = {
                "messages": [sanitazed],
                "answer": "You have rejected the addition of items to the cart."
            },
            goto = END
        )




### Edges

def product_qa_agent_tool_edge(state) -> str:
    """Decide whether to continue or end"""

    if state.product_qa_agent.final_answer:
        return "end"
    elif state.product_qa_agent.iteration > 4:
        return "end"
    elif len(state.product_qa_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"
    

def shopping_cart_agent_tool_edge(state) -> str:
    """Decide whether to continue or end"""

    add_to_cart_tool_call = False
    for tool_call in state.shopping_cart_agent.tool_calls:
        if tool_call.name == "add_to_shopping_cart":
            add_to_cart_tool_call = True
            break

    if state.shopping_cart_agent.final_answer:
        return "end"
    elif state.shopping_cart_agent.iteration > 2:
        return "end"
    elif len(state.shopping_cart_agent.tool_calls) > 0:
        if add_to_cart_tool_call:
            return "hitl_add_to_cart"
        else:
            return "tools"
    else:
        return "end"
    

def warehouse_manager_agent_tool_edge(state) -> str:
    """Decide whether to continue or end"""

    if state.warehouse_manager_agent.final_answer:
        return "end"
    elif state.warehouse_manager_agent.iteration > 2:
        return "end"
    elif len(state.warehouse_manager_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"


def coordinator_agent_edge(state) -> str:

    if state.coordinator_agent.iteration > 3:
        return "end"
    elif state.coordinator_agent.final_answer and len(state.coordinator_agent.plan) == 0:
        return "end"
    elif state.coordinator_agent.next_agent == "product_qa_agent":
        return "product_qa_agent"
    elif state.coordinator_agent.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"
    elif state.coordinator_agent.next_agent == "warehouse_manager_agent":
        return "warehouse_manager_agent"
    else:
        return "end"
    


### Workflow

workflow = StateGraph(State)

product_qa_agent_tools = [get_formatted_items_context, get_formatted_reviews_context]
product_qa_agent_tool_node = ToolNode(product_qa_agent_tools)
product_qa_agent_tool_description = get_tool_descriptions(product_qa_agent_tools)

shopping_cart_agent_tools = [add_to_shopping_cart, remove_from_cart, get_shopping_cart]
shopping_cart_agent_tool_node = ToolNode(shopping_cart_agent_tools)
shopping_cart_agent_tool_description = get_tool_descriptions(shopping_cart_agent_tools)

warehouse_manager_agent_tools = [check_warehouse_availability, reserve_warehouse_items]
warehouse_manager_agent_tool_node = ToolNode(warehouse_manager_agent_tools)
warehouse_manager_agent_tool_description = get_tool_descriptions(warehouse_manager_agent_tools)

workflow.add_node("product_qa_agent", product_qa_agent)
workflow.add_node("shopping_cart_agent", shopping_cart_agent)
workflow.add_node("warehouse_manager_agent", warehouse_manager_agent)
workflow.add_node("coordinator_agent", coordinator_agent)
workflow.add_node("hitl_add_to_cart", hitl_add_to_cart)

workflow.add_node("product_qa_agent_tool_node", product_qa_agent_tool_node)
workflow.add_node("shopping_cart_agent_tool_node", shopping_cart_agent_tool_node)
workflow.add_node("warehouse_manager_agent_tool_node", warehouse_manager_agent_tool_node)

workflow.add_edge(START, "coordinator_agent")

workflow.add_conditional_edges(
    "coordinator_agent",
    coordinator_agent_edge,
    {
        "product_qa_agent": "product_qa_agent",
        "shopping_cart_agent": "shopping_cart_agent",
        "warehouse_manager_agent": "warehouse_manager_agent",
        "end": END
    }
)

workflow.add_conditional_edges(
    "product_qa_agent",
    product_qa_agent_tool_edge,
    {
        "tools": "product_qa_agent_tool_node",
        "end": "coordinator_agent"
    }
)

workflow.add_conditional_edges(
    "shopping_cart_agent",
    shopping_cart_agent_tool_edge,
    {
        "tools": "shopping_cart_agent_tool_node",
        "hitl_add_to_cart": "hitl_add_to_cart",
        "end": "coordinator_agent"
    }
)
workflow.add_conditional_edges(
    "warehouse_manager_agent",
    warehouse_manager_agent_tool_edge,
    {
        "tools": "warehouse_manager_agent_tool_node",
        "end": "coordinator_agent"
    }
)

workflow.add_edge("product_qa_agent_tool_node", "product_qa_agent")
workflow.add_edge("shopping_cart_agent_tool_node", "shopping_cart_agent")
workflow.add_edge("warehouse_manager_agent_tool_node", "warehouse_manager_agent")




### Agent Execution

def agent_stream_wrapper(question, thread_id: str, mode: str):

    def _string_for_sse(string):
        return f"data: {string}\n\n"

    def _process_graph_event(chunk):

        def _is_interrupt(chunk):
            return len(chunk[1].get("payload", {}).get("interrupts", [])) > 0

        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        def _is_node_end(chunk):
            return chunk[0] == "updates"

        def _tool_to_text(tool_call):
            if tool_call.get("name") == "get_formatted_items_context":
                return f"Looking for items: {tool_call.get('args').get('query', '')}."
            elif tool_call.get("name") == "get_formatted_reviews_context":
                return f"Fetching user reviews..."
            else:
                return f"Unknown tool: {tool_call.name}"

        if _is_node_start(chunk):
            if chunk[1].get("payload", {}).get("name") == "coordinator_agent":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "product_qa_agent":
                return "Fetching information about inventory..."
            if chunk[1].get("payload", {}).get("name") == "shopping_cart_agent":
                return "Shopping cart management..."
            if chunk[1].get("payload", {}).get("name") == "warehouse_manager_agent":
                return "Warehouse management..."
            if chunk[1].get("payload", {}).get("name") == "tool_node":
                message = " ".join([_tool_to_text(tool_call) for tool_call in chunk[1].get('payload', {}).get('input', {}).messages[-1].tool_calls])
                return message
        elif _is_interrupt(chunk):
            value = chunk[1].get("payload", {}).get("interrupts", [])[0].get("value")
            payload = {
                "type": "hitl_interrupt",
                "data": {
                    "data": value
                }
            }
            return json.dumps(payload)
        else:
            return "Unknown operation..."

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    if mode == "initialise":
        initial_state = {
            "messages": [{"role": "user", "content": question}],
            "user_id": thread_id,
            "cart_id": thread_id,
            "product_qa_agent": {
                "iteration": 0,
                "available_tools": product_qa_agent_tool_description,
                "tool_calls": [],
                "final_answer": False
            },
            "shopping_cart_agent": {
                "iteration": 0,
                "available_tools": shopping_cart_agent_tool_description,
                "tool_calls": [],
                "final_answer": False
            },
            "warehouse_manager_agent": {
                "iteration": 0,
                "available_tools": warehouse_manager_agent_tool_description,
                "tool_calls": [],
                "final_answer": False
            },
            "coordinator_agent": {
                "iteration": 0,
                "plan": [],
                "next_agent": "", 
                "final_answer": False
            }
        }
    elif mode == "hitl":
        initial_state = Command(
            resume = {
                "confirmed": question
            }
        )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    with PostgresSaver.from_conn_string(
        "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db"
    ) as checkpointer:

        graph = workflow.compile(
            checkpointer=checkpointer
        )

        for chunk in graph.stream(
            initial_state,
            config=config,
            stream_mode=["debug", "values"]
        ):
            processed_chunk = _process_graph_event(chunk)

            if processed_chunk:
                yield _string_for_sse(processed_chunk)

            if chunk[0] == "values":
                result = chunk[1]

    used_context = []
    dummy_vector = np.zeros(1536)

    for item in result.get("references", []):
        payload = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            query=dummy_vector,
            using="text-embedding-3-small",
            limit=1,
            with_payload=True,
            with_vectors=False,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id)
                    )
                ]
            )
        ).points[0].payload
        image_url = payload.get("image")
        price = payload.get("price")
        if image_url:
            used_context.append(
                {
                    "image_url": image_url,
                    "price": price,
                    "description": item.description
                }
            )

    shopping_cart = get_shopping_cart(thread_id, thread_id)
    shopping_cart_items = [
        {
            "price": float(item.get("price")) if item.get("price") else None,
            "quantity": item.get("quantity"),
            "currency": item.get("quantity"),
            "product_image_url": item.get("product_image_url"),
            "total_price": float(item.get("total_price")) if item.get("total_price") else None,
        }
        for item in shopping_cart
    ]

    yield _string_for_sse(json.dumps(
        {
            "type": "final_answer",
            "data": {
                "answer": result.get("answer", ""),
                "used_context": used_context,
                "trace_id": result.get("trace_id", ""),
                "shopping_cart": shopping_cart_items
            }
        }
    ))
