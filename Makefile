dc-up:
	uv sync
	docker compose up --build

dc-down:
	docker compose down

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb