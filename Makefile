.PHONY: extract-pdf extract-entities extract app

extract-pdf:
	uv run python -m src.extraction.pdf_extractor

extract-entities:
	uv run python -m src.extraction.entity_extractor

extract: extract-pdf extract-entities

app:
	uv run python -m app.app
