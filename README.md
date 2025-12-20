# Project Setup and Usage

## Prerequisites
- Python 3.9+ recommended
- pip and virtualenv (or conda) available
- Access to Google Vertex AI Gemini API

## Install Dependencies
Run the following from the project root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers tqdm numpy requests
```
If you plan to use a GPU build of PyTorch, install the matching wheel from the official PyTorch instructions instead of the CPU default above.

## Environment Variables
The script now reads the Gemini API key from an environment variable:
- `GEMINI_API_KEY`: your Vertex AI Gemini API key

Example:
```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

## Data Layout
Ensure the `Amazon_products` directory and its contents (classes, hierarchy, train/test corpus, etc.) remain in place relative to `real_final_code.py`. The script writes cached artifacts and outputs to `Amazon_products/_artifacts_roberta` and produces `submission.csv` in the project root.

## Running
With the virtual environment active and `GEMINI_API_KEY` set:
```bash
python real_final_code.py
```
The script will cache intermediate artifacts so reruns resume quickly. API calls are limited by the `API_CALL_LIMIT` constant in the script.
