.PHONY: tests run streamlit dashboard fake-data online-judge

run:
	uv run python main.py

streamlit:
	uv run streamlit run app.py

dashboard:
	uv run streamlit run dashboard.py --server.port 8502

tests:
	uv run pytest

tests-judge:
	uv run pytest tests/test_judge.py -n 4

fake-data:
	DATABASE_URL="postgresql://alexey:admin@localhost:5432/logs" uv run python -m logs.generate_fake_data --rate 2 $(ARGS)

fake-data-sqlite:
	uv run python -m logs.generate_fake_data --rate 2 --db db/logs.db $(ARGS)

fake-data-docker:
	docker compose exec streamlit-app uv run python -m logs.generate_fake_data --rate 2 $(ARGS)

diag-db:
	docker compose exec streamlit-app uv run python diag_db.py

online-judge:
	uv run python -m logs.judge.main
