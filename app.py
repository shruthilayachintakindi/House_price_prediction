import csv
import math
import random
import statistics
from collections import Counter
from pathlib import Path

import streamlit as st


DATA_PATH = Path(__file__).with_name("Housing.csv")
TARGET = "price"
NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
CATEGORICAL_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]


@st.cache_data
def load_data():
    with DATA_PATH.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        data = list(reader)

    expected_columns = {TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
    missing_columns = expected_columns.difference(reader.fieldnames or [])
    if missing_columns:
        raise ValueError(f"Housing.csv is missing columns: {', '.join(sorted(missing_columns))}")

    for row in data:
        row[TARGET] = float(row[TARGET])
        for feature in NUMERIC_FEATURES:
            row[feature] = float(row[feature])
    return data


def feature_scales(records):
    return {
        feature: max(
            max(record[feature] for record in records)
            - min(record[feature] for record in records),
            1.0,
        )
        for feature in NUMERIC_FEATURES
    }


def predict_price(home, records, scales):
    distances = []
    for record in records:
        distance = sum(
            ((home[feature] - record[feature]) / scales[feature]) ** 2
            for feature in NUMERIC_FEATURES
        )
        distance += sum(
            home[feature] != record[feature] for feature in CATEGORICAL_FEATURES
        )
        distances.append((distance, record[TARGET]))

    nearest = sorted(distances, key=lambda item: item[0])[: min(7, len(distances))]
    exact_matches = [price for distance, price in nearest if distance == 0]
    if exact_matches:
        return statistics.mean(exact_matches)

    weighted_prices = [price / math.sqrt(distance) for distance, price in nearest]
    weights = [1 / math.sqrt(distance) for distance, _ in nearest]
    return sum(weighted_prices) / sum(weights)


@st.cache_data
def evaluate_model(records):
    shuffled_records = list(records)
    random.Random(42).shuffle(shuffled_records)
    test_count = max(1, round(len(shuffled_records) * 0.2))
    test_records = shuffled_records[:test_count]
    training_records = shuffled_records[test_count:]
    scales = feature_scales(training_records)
    actual = [record[TARGET] for record in test_records]
    predicted = [predict_price(record, training_records, scales) for record in test_records]
    mae = statistics.mean(abs(expected - estimate) for expected, estimate in zip(actual, predicted))
    mean_actual = statistics.mean(actual)
    total_variation = sum((value - mean_actual) ** 2 for value in actual)
    residual_variation = sum((expected - estimate) ** 2 for expected, estimate in zip(actual, predicted))
    r2 = 1 - residual_variation / total_variation if total_variation else 0.0
    return {"mae": mae, "r2": r2}


st.set_page_config(page_title="House Price Predictor", page_icon="⌂", layout="centered")
st.title("House Price Predictor")
st.caption("Estimate a sale price from the home's size, features, and furnishing status.")

try:
    housing_data = load_data()
    if len(housing_data) < 2:
        raise ValueError("Housing.csv must contain at least two data rows.")
    model_metrics = evaluate_model(housing_data)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

metric_columns = st.columns(2)
metric_columns[0].metric("Test-set mean absolute error", f"{model_metrics['mae']:,.0f}")
metric_columns[1].metric("Test-set R²", f"{model_metrics['r2']:.2f}")

numeric_defaults = {
    feature: round(statistics.median(record[feature] for record in housing_data))
    for feature in NUMERIC_FEATURES
}
categorical_defaults = {
    feature: Counter(record[feature] for record in housing_data).most_common(1)[0][0]
    for feature in CATEGORICAL_FEATURES
}

with st.form("prediction_form"):
    st.subheader("Property details")
    first_row = st.columns(3)
    area = first_row[0].number_input(
        "Area (sq ft)", min_value=1, value=max(1, int(numeric_defaults["area"])), step=100
    )
    bedrooms = first_row[1].number_input(
        "Bedrooms", min_value=0, value=int(numeric_defaults["bedrooms"]), step=1
    )
    bathrooms = first_row[2].number_input(
        "Bathrooms", min_value=0, value=int(numeric_defaults["bathrooms"]), step=1
    )

    second_row = st.columns(2)
    stories = second_row[0].number_input(
        "Stories", min_value=1, value=max(1, int(numeric_defaults["stories"])), step=1
    )
    parking = second_row[1].number_input(
        "Parking spaces", min_value=0, value=int(numeric_defaults["parking"]), step=1
    )

    categorical_inputs = {}
    for start in range(0, len(CATEGORICAL_FEATURES), 2):
        fields = st.columns(2)
        for offset, feature in enumerate(CATEGORICAL_FEATURES[start : start + 2]):
            options = sorted({record[feature] for record in housing_data})
            default = categorical_defaults[feature]
            categorical_inputs[feature] = fields[offset].selectbox(
                feature.replace("_", " ").title(),
                options,
                index=options.index(default),
            )

    submitted = st.form_submit_button("Estimate price", type="primary", use_container_width=True)

if submitted:
    property_details = {
        "area": float(area),
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "stories": float(stories),
        "parking": float(parking),
        **categorical_inputs,
    }
    estimate = predict_price(property_details, housing_data, feature_scales(housing_data))
    st.success(f"Estimated price: {estimate:,.0f}")