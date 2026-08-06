# Pipeline Corrosion AI

> FDE Cross-Industry Portfolio | Project 1 | Energy Sector
> Solving corrosion prediction and standards retrieval problems in pipeline integrity management with AI.

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pipeline-corrosion-ai.streamlit.app/)

## Problem Statement

- **Industry**: Oil & gas pipeline integrity management
- **Pain Point 1**: Corrosion risk prediction relies on engineer experience, historical data underutilized
- **Pain Point 2**: NACE/API/ASME standards retrieval takes 15-30 minutes per query
- **AI Solution**: ML prediction model + RAG intelligent Q&A

## Features

### Tab 1: Corrosion Prediction
Input pipeline parameters (material, temperature, pH, CO2 partial pressure, H2S concentration, flow rate, chloride content) -> predict corrosion rate and risk level -> provide protection recommendations

### Tab 2: Standards Q&A
Intelligent Q&A on NACE MR0175, API 571, ASME B31.8S and other standards with natural language queries

## Tech Stack

| Component | Technology |
|-----------|------------|
| Web UI | Streamlit |
| Prediction Model | scikit-learn GradientBoosting (R2=0.80) |
| RAG Engine | Dify Cloud Streaming SSE + LRU Cache |
| Deployment | Streamlit Community Cloud |

## Quick Start

```bash
git clone https://github.com/phdleo101/pipeline-corrosion-ai.git
cd pipeline-corrosion-ai
python -m venv venv
source venv/bin/activate  # Linux/Mac | venv\Scripts\activate (Windows)
pip install -r requirements.txt
python src/data_processor.py
python src/corrosion_model.py
streamlit run src/streamlit_app.py
```

Visit http://localhost:8501

## License

MIT
