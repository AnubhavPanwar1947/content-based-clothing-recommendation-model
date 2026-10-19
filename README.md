# Content-Based Clothing Recommendation Model

This project builds an item-to-item clothing recommender using product
attributes such as category, color, style, material, gender, and season.

## Project contents

- `data/products.json` — clothing catalog
- `notebooks/01_content_based_recommender.ipynb` — data preparation,
  TF-IDF feature creation, cosine similarity, and recommendations
- `requirements.txt` — Python dependencies

## Setup on another computer

1. Install Python 3.11 or 3.12.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install the dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Start Jupyter:

   ```powershell
   python -m jupyter notebook
   ```

6. Open `notebooks/01_content_based_recommender.ipynb` and select the
   `.venv` Python kernel.

The `.venv` folder is intentionally excluded from Git. It is recreated
locally from `requirements.txt`.
