.PHONY: test eval build-index app

test:
	pytest -q

eval:
	python -m src.evaluate_retrieval

build-index:
	python -m src.build_index

app:
	streamlit run app.py
