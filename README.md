# PSR

## 1. Requirements

### 1.1 Python Environment

- Python >= 3.10

### 1.2 Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

<details>
<summary>Key dependencies</summary>

| Package | Version |
|---------|---------|
| numpy | 1.26.4 |
| tqdm | 4.66.4 |
| requests | 2.32.3 |
| openai | 1.40.6 |
| SPARQLWrapper | 2.0.0 |
| sentence-transformers | 3.0.1 |
| transformers | 4.44.2 |
| tokenizers | 0.19.1 |

</details>

### 1.3 External Services

| Service | Endpoint | Description |
|---------|----------|-------------|
| Virtuoso SPARQL | `http://localhost:8890/sparql` | Must be loaded with a Freebase-compatible RDF dump using namespace `http://rdf.freebase.com/ns/` |

## 2. Project Structure

```
├── main.py              # End-to-end pipeline and evaluation
├── run.py               # Entry point script
├── llm_client.py        # LLM API interface with retry logic
├── freebase_func.py     # SPARQL utilities for Freebase (incl. CVT handling)
├── chains_oneshot.py    # One-shot schema-level path enumeration
├── schema_tri.py        # Schema trie construction and management
├── prompt_list.py       # Prompt templates for LLM calls
├── freebase_schema.csv  # Freebase schema definitions
├── described.jsonl      # Entity descriptions
├── data/                # Input datasets
├── output/              # Intermediate results
├── requirements.txt
└── README.md
```

## 3. Dataset Format

The input dataset is a JSON file containing a list of question objects:

```json
[
  {
    "question": "Who directed Titanic?",
    "topic_entity": {
      "m.0jcx": "Titanic"
    },
    "answer": "James Cameron"
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `question` | string | Natural language question |
| `topic_entity` | dict | Mapping from Freebase MIDs to entity names |
| `answer` | string | Gold answer |

## 4. Configuration

Set the following environment variables before running:

```bash
export LLM_API_KEY=<your_api_key>
export LLM_API_BASE=<your_api_base_url>
```

## 5. Usage

```bash
python main.py \
    --data_path data/cwq.json \
    --limit 100 \
    --workers 8 \
    --model gpt-4
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to input dataset | - |
| `--limit` | Number of questions to evaluate | `None` (all) |
| `--workers` | Number of parallel workers | `1` |
| `--model` | LLM backend identifier | `gpt-4` |

## 6. Output

### 6.1 Predictions

Final predictions are saved in JSON format:

```json
{
  "idx": 0,
  "question": "...",
  "gold": ["James Cameron"],
  "pred": "James Cameron"
}
```

### 6.2 Intermediate Results

Per-question artifacts are saved under `output/` for analysis and debugging.

## License

This project is released for research purposes.
