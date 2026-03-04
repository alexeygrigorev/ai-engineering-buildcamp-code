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
	uv run python -m logs.generate_fake_data --rate 2

online-judge:
	uv run python -m logs.judge.main
