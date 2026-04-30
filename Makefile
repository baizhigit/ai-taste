dc-up:
	uv sync
	docker compose up --build

dc-down:
	docker compose down

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb

run-evals-retriever:
	uv sync
	PYTHONPATH=${PWD}/apps/api:${PWD}/apps/api/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.eval_retriever