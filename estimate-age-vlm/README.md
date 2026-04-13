# Deer Age Estimation

A demo app that estimates deer age from jaw/teeth images using OpenAI vision via LangChain.

## Stack

- **API**: FastAPI + LangChain + OpenAI (`gpt-5.4-mini`)
- **UI**: Streamlit

## Project Structure

```
deer-age-estimation/
├── api/
│   ├── app/
│   │   ├── routers/estimate.py   # POST /api/v1/estimate
│   │   ├── services/estimator.py # LangChain vision chain
│   │   ├── schemas.py            # Request/response models
│   │   ├── config.py             # Settings
│   │   └── main.py
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── ui/
│   ├── utils/api_client.py       # HTTP client for the API
│   ├── app.py                    # Streamlit app
│   ├── requirements.txt
│   └── .env.example
└── .gitignore
```

## Requirements

- Python >= 3.10

## Setup

### API

```bash
cd api
python -m venv .venv && source .venv/bin/activate || .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # fill in your OPENAI_API_KEY
uvicorn app.main:app --reload
```

### UI

```bash
cd ui
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```
