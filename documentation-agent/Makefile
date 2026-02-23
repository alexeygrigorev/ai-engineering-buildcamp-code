.PHONY: tests run streamlit

run:
	uv run python main.py

streamlit:
	uv run streamlit run app.py

tests:
	uv run pytest

tests-judge:
	uv run pytest tests/test_judge.py -n 4
