Overview
- Input example: `2 cups rice, 1 tbsp oil, 200g chicken`
- Output example: `~600 kcal, 30g protein, 20g fat, 70g carbs`


This repo contains:
- `model_train.py` — training script (preprocessing + model)
- `utils/preprocess.py` — ingredient parser & helper functions
- `app.py` — Flask backend that loads saved model/tokenizer and serves `/predict`
- `frontend_streamlit.py` — Streamlit frontend to input ingredient lists and display results


## Quickstart (local)
1. Create virtual environment and install requirements:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
