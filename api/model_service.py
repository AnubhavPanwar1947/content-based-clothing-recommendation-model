import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load the catalog
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "products.json"

with DATA_PATH.open("r", encoding="utf-8") as file:
    products = json.load(file)

products_df = pd.DataFrame(products)

required_columns = {
    "id",
    "name",
    "category",
    "gender",
    "colors",
    "style",
    "material",
    "season",
    "price",
    "stock",
}

missing_columns = required_columns - set(products_df.columns)

if missing_columns:
    raise ValueError(
        f"Missing catalog columns: {sorted(missing_columns)}"
    )

if not products_df["id"].is_unique:
    raise ValueError("Product IDs must be unique")


# -----------------------------
# Create product features
# -----------------------------

feature_columns = [
    "category",
    "gender",
    "colors",
    "style",
    "material",
    "season",
]


def build_feature_text(row):
    parts = []

    for column in feature_columns:
        value = row[column]

        if isinstance(value, list):
            parts.extend(
                str(item).casefold()
                for item in value
            )
        else:
            parts.append(str(value).casefold())

    return " ".join(parts)


products_df["feature_text"] = products_df.apply(
    build_feature_text,
    axis=1
)


# -----------------------------
# Recommendation vectors
# -----------------------------

vectorizer = TfidfVectorizer()

feature_matrix = vectorizer.fit_transform(
    products_df["feature_text"]
)

similarity_matrix = cosine_similarity(feature_matrix)


# -----------------------------
# Search vectors
# -----------------------------

products_df["search_text"] = (
    products_df["name"].fillna("")
    + " "
    + products_df["feature_text"]
)

search_vectorizer = TfidfVectorizer()

search_matrix = search_vectorizer.fit_transform(
    products_df["search_text"]
)


# -----------------------------
# Search helpers
# -----------------------------

def normalize_search_query(query):
    query = str(query).casefold()
    query = re.sub(r"\s+", " ", query).strip()

    synonyms = {
        "t shirt": "t-shirt",
        "tshirt": "t-shirt",
        "tee": "t-shirt",
        "tees": "t-shirt",
        "pant": "pants",
        "trouser": "trousers",
        "jean": "jeans",
        "hoodies": "hoodie",
        "shirts": "shirt",
        "jackets": "jacket",
        "sweaters": "sweater",
        "dresses": "dress",
        "dark blue": "dark-blue",
        "light blue": "light-blue",
        "navy blue": "navy",
    }

    for original, standard in sorted(
        synonyms.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        query = re.sub(
            rf"(?<![a-z0-9]){re.escape(original)}(?![a-z0-9])",
            standard,
            query,
        )

    return re.sub(r"\s+", " ", query).strip()


def _column_values(column_name):
    values = set()

    for value in products_df[column_name].dropna():
        if isinstance(value, list):
            values.update(
                str(item).casefold().strip()
                for item in value
            )
        else:
            values.add(str(value).casefold().strip())

    return {value for value in values if value}


def _query_contains(query, phrase):
    return re.search(
        rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
        query,
    ) is not None


def _row_contains_any(values, requested_values):
    if not isinstance(values, list):
        values = [values]

    actual_values = {
        str(value).casefold().strip()
        for value in values
    }

    return bool(actual_values.intersection(requested_values))


def extract_query_features(query):
    normalized_query = normalize_search_query(query)

    category_aliases = {
        "pants": "pants",
        "jeans": "jeans",
        "trousers": "trousers",
        "t-shirt": "t-shirt",
        "shirt": "shirt",
        "hoodie": "hoodie",
        "sweater": "sweater",
        "jacket": "jacket",
        "dress": "dress",
        "lounge-set": "lounge-set",
    }

    category = None

    for alias in sorted(
        category_aliases,
        key=len,
        reverse=True,
    ):
        if _query_contains(normalized_query, alias):
            category = category_aliases[alias]
            break

    known_colors = _column_values("colors")
    known_styles = _column_values("style")
    known_materials = _column_values("material")
    known_seasons = _column_values("season")
    known_genders = _column_values("gender")

    colors = {
        value
        for value in known_colors
        if _query_contains(normalized_query, value)
    }

    styles = {
        value
        for value in known_styles
        if _query_contains(normalized_query, value)
    }

    materials = {
        value
        for value in known_materials
        if _query_contains(normalized_query, value)
    }

    seasons = {
        value
        for value in known_seasons
        if _query_contains(normalized_query, value)
    }

    genders = {
        value
        for value in known_genders
        if _query_contains(normalized_query, value)
    }

    price_match = re.search(
        r"(?:under|below|less than|up to|upto)\s*"
        r"(?:₹|rs\.?|inr|\$)?\s*([\d,]+)",
        normalized_query,
    )

    max_price = (
        int(price_match.group(1).replace(",", ""))
        if price_match
        else None
    )

    return {
        "normalized_query": normalized_query,
        "category": category,
        "colors": colors,
        "styles": styles,
        "materials": materials,
        "seasons": seasons,
        "genders": genders,
        "max_price": max_price,
    }


# -----------------------------
# Search function
# -----------------------------

def search_products(
    query,
    n=50,
    gender=None,
    include_unisex=True,
):
    if search_matrix.shape[0] != len(products_df):
        raise RuntimeError(
            "Catalog and search matrix are out of sync. "
            "Restart the model and rebuild the data."
        )

    n = int(n)

    if n <= 0:
        return pd.DataFrame()

    features = extract_query_features(query)
    normalized_query = features["normalized_query"]

    query_vector = search_vectorizer.transform(
        [normalized_query]
    )

    if query_vector.nnz > 0:
        scores = cosine_similarity(
            query_vector,
            search_matrix,
        ).ravel()
    else:
        scores = np.zeros(len(products_df))

    stock_values = pd.to_numeric(
        products_df["stock"],
        errors="coerce",
    ).fillna(0)

    in_stock = stock_values.gt(0)
    exact_mask = in_stock.copy()
    has_structured_filter = False

    category_groups = {
        "pants": {"jeans", "trousers", "pants"},
        "jeans": {"jeans"},
        "trousers": {"trousers"},
        "t-shirt": {"t-shirt", "t-shirts"},
        "shirt": {"shirt", "shirts"},
        "hoodie": {"hoodie", "hoodies"},
        "sweater": {"sweater", "sweaters"},
        "jacket": {"jacket", "jackets"},
        "dress": {"dress", "dresses"},
        "lounge-set": {"lounge-set", "lounge set"},
    }

    category_values = (
        products_df["category"]
        .astype(str)
        .str.casefold()
        .str.strip()
    )

    if features["category"]:
        exact_mask &= category_values.isin(
            category_groups[features["category"]]
        )
        has_structured_filter = True

    for column_name, requested_values in [
        ("colors", features["colors"]),
        ("style", features["styles"]),
        ("material", features["materials"]),
        ("season", features["seasons"]),
    ]:
        if requested_values:
            exact_mask &= products_df[column_name].apply(
                lambda values: _row_contains_any(
                    values,
                    requested_values,
                )
            )
            has_structured_filter = True

    gender_values = (
        products_df["gender"]
        .astype(str)
        .str.casefold()
        .str.strip()
    )

    if gender is not None:
        requested_gender = str(gender).casefold().strip()
        allowed_genders = {requested_gender}

        if include_unisex and requested_gender in {
            "men",
            "women",
        }:
            allowed_genders.add("unisex")

        exact_mask &= gender_values.isin(allowed_genders)
        has_structured_filter = True

    elif features["genders"]:
        exact_mask &= gender_values.isin(features["genders"])
        has_structured_filter = True

    if features["max_price"] is not None:
        exact_mask &= products_df["price"].fillna(
            np.inf
        ).le(features["max_price"])

        has_structured_filter = True

    if has_structured_filter and exact_mask.any():
        candidate_mask = exact_mask
        match_type = "exact"
    else:
        candidate_mask = in_stock & (scores > 0)
        match_type = "related"

        if gender is not None:
            candidate_mask &= gender_values.isin(
                allowed_genders
            )

        if features["max_price"] is not None:
            candidate_mask &= products_df["price"].fillna(
                np.inf
            ).le(features["max_price"])

    if not candidate_mask.any():
        return pd.DataFrame()

    results = products_df.loc[candidate_mask].copy()

    results["search_score"] = scores[
        candidate_mask.to_numpy()
    ]

    results["match_type"] = match_type

    results = results.sort_values(
        ["search_score", "popularity"],
        ascending=[False, False],
    )

    return results[
        [
            "id",
            "name",
            "category",
            "gender",
            "colors",
            "style",
            "price",
            "stock",
            "search_score",
            "match_type",
        ]
    ].head(n).reset_index(drop=True)


# -----------------------------
# Recommendation function
# -----------------------------

def _canonical_category(value):
    value = str(value).casefold().strip()

    aliases = {
        "tshirt": "t-shirt",
        "t-shirts": "t-shirt",
        "t shirt": "t-shirt",
        "shirts": "shirt",
        "hoodies": "hoodie",
        "sweaters": "sweater",
        "jackets": "jacket",
        "dresses": "dress",
        "trouser": "trousers",
        "pants": "pants",
        "jean": "jeans",
    }

    return aliases.get(value, value)


def recommend_products(
    product_id,
    n=5,
    gender=None,
    category=None,
    include_unisex=True,
):
    product_id = str(product_id).strip()
    product_ids = products_df["id"].astype(str).tolist()

    if product_id not in product_ids:
        raise ValueError(
            f"Product ID not found: {product_id}"
        )

    n = int(n)

    if n <= 0:
        return pd.DataFrame()

    product_position = product_ids.index(product_id)
    selected_product = products_df.iloc[product_position]

    category_groups = {
        "pants": {"pants", "trousers", "jeans"},
        "jeans": {"jeans"},
        "trousers": {"trousers"},
        "t-shirt": {"t-shirt", "t-shirts"},
        "shirt": {"shirt", "shirts"},
        "hoodie": {"hoodie", "hoodies"},
        "sweater": {"sweater", "sweaters"},
        "jacket": {"jacket", "jackets"},
        "dress": {"dress", "dresses"},
        "lounge-set": {"lounge-set", "lounge set"},
    }

    requested_category = _canonical_category(
        category
        if category is not None
        else selected_product["category"]
    )

    allowed_categories = category_groups.get(
        requested_category,
        {requested_category},
    )

    category_values = (
        products_df["category"]
        .astype(str)
        .str.casefold()
        .str.strip()
    )

    stock_values = pd.to_numeric(
        products_df["stock"],
        errors="coerce",
    ).fillna(0)

    candidate_mask = (
        category_values.isin(allowed_categories)
        & stock_values.gt(0)
    )

    if gender is not None:
        requested_gender = str(gender).casefold().strip()
        allowed_genders = {requested_gender}

        if include_unisex and requested_gender in {
            "men",
            "women",
        }:
            allowed_genders.add("unisex")

        gender_values = (
            products_df["gender"]
            .astype(str)
            .str.casefold()
            .str.strip()
        )

        candidate_mask &= gender_values.isin(
            allowed_genders
        )

    candidate_mask.iloc[product_position] = False

    if not candidate_mask.any():
        return pd.DataFrame()

    scores = similarity_matrix[product_position]

    ranked_positions = [
        int(position)
        for position in scores.argsort()[::-1]
        if candidate_mask.iloc[int(position)]
    ][:n]

    recommendations = products_df.iloc[ranked_positions][
        [
            "id",
            "name",
            "category",
            "gender",
            "colors",
            "style",
            "price",
            "stock",
        ]
    ].copy()

    recommendations["similarity_score"] = (
        scores[ranked_positions].round(3)
    )

    return recommendations.reset_index(drop=True)