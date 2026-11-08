PORT ?= 8830

.PHONY: install test lint format run eval gate screenshot docker-build clean

install:
	uv sync --python 3.12

test:
	uv run --python 3.12 --group dev pytest -q

lint:
	uvx ruff check .

format:
	uvx ruff format .
	uvx ruff check . --fix

run:  ## trains on first start (about 30 seconds), then loads data/study.pkl
	PORT=$(PORT) uv run uplift serve

eval:  ## full study, writes reports/eval.json and reports/summary.md, caches the fitted study for the server
	uv run uplift eval --save data/study.pkl

gate:  ## exit 1 unless the best uplift policy's true net value beats churn risk targeting
	uv run uplift eval --gate --out reports-ci

screenshot:  ## needs the server running on $(PORT) and Google Chrome installed (macOS path)
	"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars \
		--virtual-time-budget=10000 --window-size=1440,1960 --screenshot=docs/screenshot.png \
		"http://127.0.0.1:$(PORT)/?customer=C000293"

docker-build:
	docker build -t uplift-retention .

clean:
	rm -rf .pytest_cache .ruff_cache reports-ci data
