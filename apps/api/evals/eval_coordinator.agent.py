from api.agents.agents import coordinator_agent
from api.agents.graph import State

from langsmith import Client
from time import sleep


ACC_THRESHOLD = 0.9
SLEEP_TIME = 10

ls_client = Client()


def next_agent_evaluator(run, example):

    next_agent_match = run.outputs["coordinator_agent"]["next_agent"] == example.outputs["next_agent"]
    final_answer_match = run.outputs["coordinator_agent"]["final_answer"] == example.outputs["coordinator_final_answer"]

    return next_agent_match and final_answer_match


results = ls_client.evaluate(
    lambda x: coordinator_agent(State(messages=x["messages"])),
    data="coordinator_eval_dataset",
    # num_repetitions=2,
    max_concurrency=10,
    evaluators=[
        next_agent_evaluator
    ],
    experiment_prefix="coordinator-eval-dataset"
)

print(f"Sleeping for {SLEEP_TIME} seconds...")
sleep(SLEEP_TIME)

results_resp = ls_client.read_project(
    project_name=results.experiment_name,
    include_stats=True
)
feedback_stats_exist = results_resp.feedback_stats.get("next_agent_evaluator") is not None

while not feedback_stats_exist:

    results_resp = ls_client.read_project(
        project_name=results.experiment_name,
        include_stats=True
    )
    feedback_stats_exist = results_resp.feedback_stats.get("next_agent_evaluator") is not None


avg_metric = results_resp.feedback_stats.get("next_agent_evaluator").get("avg")
errors = results_resp.feedback_stats.get("next_agent_evaluator").get("errors")


if avg_metric >= ACC_THRESHOLD:
    output_message = f"✅ Success: {avg_metric}"
else:
    output_message = f"❌ Success: {avg_metric}"


if errors > 0:
    raise AssertionError(f"Evaluation failed with {errors} errors")
elif avg_metric >= ACC_THRESHOLD:
    print(output_message, flush=True)
else:
    raise AssertionError(output_message)
