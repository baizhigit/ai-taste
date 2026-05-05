from pydantic import BaseModel, Field
from langsmith import traceable, get_current_run_tree
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
import instructor
from langchain_core.messages import convert_to_openai_messages

from api.agents.tools import get_formatted_item_context
from api.agents.utils.prompt_management import prompt_template_config


### QnA Agent Response Model

class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class FinalResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")


### Intent Router Response Model

class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str


### QnA Agent Node

@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def agent_node(state) -> dict:

    template = prompt_template_config("api/agents/prompts/qa_agent.yaml", "qa_agent")
    prompt = template.render()

    messages = state.messages

    llm = ChatOpenAI(model="gpt-4.1-mini")
    llm_with_tools = llm.bind_tools(
        [get_formatted_item_context, FinalResponse],
        tool_choice="auto"
    )

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=prompt),
            *messages
        ]
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.response_metadata["token_usage"]["prompt_tokens"],
            "output_tokens": response.response_metadata["token_usage"]["completion_tokens"],
            "total_tokens": response.response_metadata["token_usage"]["total_tokens"]
        }

    final_answer = False
    answer = ""
    references = []

    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get("name") == "FinalResponse":
                final_answer = True
                answer = tool_call.get("args").get("answer")
                references.extend(tool_call.get("args").get("references"))

    return {
        "messages": [response],
        "iteration": state.iteration + 1,
        "answer": answer,
        "final_answer": final_answer,
        "references": references
    }


### Intent Router Node

@traceable(
    name="route_intent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def intent_router_node(state) -> dict:

    template = prompt_template_config("api/agents/prompts/intent_router_agent.yaml", "intent_router_agent")
    prompt = template.render()

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS
    )

    response, raw_response = client.create_with_completion(
        messages=[
            {"role": "system", "content": prompt},
            *conversation
        ],
        response_model=IntentRouterResponse
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.input_tokens,
            "output_tokens": raw_response.usage.output_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }
 
    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer
    }