from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_used_car_price_model.pkl"
OPTIONS_PATH = ROOT / "car_options.json"
DATE_MIN = date(2019, 2, 20)
DATE_MAX = date(2021, 5, 30)
EXPECTED_FEATURES = [
    "fuel_type",
    "kms_run",
    "city",
    "body_type",
    "transmission",
    "variant",
    "assured_buy",
    "registered_city",
    "registered_state",
    "rto",
    "source",
    "make",
    "model",
    "total_owners",
    "fitness_certificate",
    "warranty_avail",
    "ad_year",
    "ad_month",
    "ad_dayofweek",
    "car_age",
]


st.set_page_config(
    page_title="Roadwise | Used Car Valuation",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --ink:#16283b; --muted:#66788a; --teal:#087e78; --line:#e3eaf0; }
      .stApp { background: #f4f7f9; color: var(--ink); color-scheme: light; }
      [data-testid="stHeader"] { background: rgba(244,247,249,0.88); }
      .block-container { max-width: 1420px; padding: 2rem 2rem 4rem; }
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.025em; }
      .hero { padding: 30px 34px; border-radius: 22px; color: white;
        background: radial-gradient(circle at 88% 8%, rgba(94,211,189,.34), transparent 28%),
                    linear-gradient(118deg, #142a3f 0%, #1b4554 58%, #147a72 100%);
        box-shadow: 0 16px 36px rgba(23,49,69,.13); margin-bottom: 22px; }
      .hero-kicker { color: #a7e8d9; font-size: .76rem; font-weight: 700;
        letter-spacing: .14em; text-transform: uppercase; margin: 0 0 8px; }
      .hero-title { color: #fff; font: 800 2.15rem/1.12 'Manrope', sans-serif;
        letter-spacing: -.04em; margin: 0 0 8px; }
      .hero-copy { color: #d4e5e8; font-size: 1rem; margin: 0; max-width: 720px; }
      .section-label { color:#087e78; font-size:.72rem; font-weight:700;
        letter-spacing:.13em; text-transform:uppercase; margin:0 0 2px; }
      .section-title { color:#16283b; font:700 1.15rem 'Manrope',sans-serif;
        margin:0 0 12px; }
      .microcopy { color:#718194; font-size:.86rem; }
      .price-value { color:#087e78; font:800 clamp(2.25rem,4vw,3.55rem)/1.1 'Manrope',sans-serif;
        letter-spacing:-.045em; margin:.15rem 0 .4rem; }
      .result-label { color:#687d8d; font-size:.73rem; font-weight:700;
        letter-spacing:.13em; text-transform:uppercase; }
      .result-tag { display:inline-block; color:#087e78; background:#e8f6f2;
        border-radius:999px; padding:5px 10px; font-size:.75rem; font-weight:700; }
      div[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.9);
        border-color:#e3eaf0 !important; border-radius:16px !important; }
      div.stButton > button[kind="primary"] { background:#087e78; border:0;
        border-radius:10px; min-height:48px; font-weight:700; }
      div.stButton > button[kind="primary"]:hover { background:#066c67; border:0; }
      [data-testid="stMetric"] { background:#f7fafb; border:1px solid #e8eef2;
        padding:12px 14px; border-radius:12px; }
      [data-testid="stMetricLabel"] { color:#66788a; }
      [data-testid="stMetricValue"] { color:#16283b; font-family:'Manrope',sans-serif; }
      [data-testid="stWidgetLabel"] p { color:#405568 !important; }
      [data-baseweb="select"] > div { background:#fff !important; color:#16283b !important;
        border-color:#d8e2e9 !important; }
      [data-baseweb="select"] input { color:#16283b !important; }
      .stCaption { color:#718194; }
      hr { border-color:#e7edf1; }
      footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the valuation model…")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_vehicle_options():
    with OPTIONS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def preferred(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def signature_for(row: dict) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)


try:
    model = load_model()
    vehicle_options = load_vehicle_options()
except Exception as exc:  # Show a clear local setup error instead of a blank page.
    st.error("The app could not load its model files. Check that the model and car_options.json are beside app.py.")
    st.code(str(exc))
    st.stop()


try:
    preprocessor = model.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"]
    categorical_columns = next(
        columns for name, _transformer, columns in preprocessor.transformers_ if name == "cat"
    )
    category_options = {
        column: [str(value) for value in values]
        for column, values in zip(categorical_columns, encoder.categories_)
    }
    feature_names = list(model.feature_names_in_)
    if feature_names != EXPECTED_FEATURES or not hasattr(model, "predict"):
        raise ValueError("The supplied model schema does not match this app's expected 20 input fields.")
except Exception as exc:
    st.error("The loaded model does not have the expected project pipeline structure.")
    st.code(str(exc))
    st.stop()


make_options = [
    make for make in sorted(vehicle_options)
    if make in category_options.get("make", [])
]
model_category_set = set(category_options.get("model", []))
variant_category_set = set(category_options.get("variant", []))


def models_for(make: str) -> list[str]:
    return [
        model_name for model_name in vehicle_options.get(make, {})
        if model_name in model_category_set
    ]


def variants_for(make: str, model_name: str) -> list[str]:
    return [
        variant for variant in vehicle_options.get(make, {}).get(model_name, [])
        if variant in variant_category_set
    ]


def change_make() -> None:
    models = models_for(st.session_state.make)
    if models:
        st.session_state.car_model = models[0]
        variants = variants_for(st.session_state.make, models[0])
        st.session_state.variant = variants[0] if variants else ""


def change_model() -> None:
    variants = variants_for(st.session_state.make, st.session_state.car_model)
    if variants:
        st.session_state.variant = variants[0]


if not make_options:
    st.error("No matching vehicle makes were found in the saved model. Check car_options.json.")
    st.stop()

default_make = "maruti" if "maruti" in make_options else make_options[0]
if "make" not in st.session_state or st.session_state.make not in make_options:
    st.session_state.make = default_make
initial_models = models_for(st.session_state.make)
if not initial_models:
    st.error("No trained model choices were found for this make.")
    st.stop()
if "car_model" not in st.session_state or st.session_state.car_model not in initial_models:
    st.session_state.car_model = "swift" if "swift" in initial_models else initial_models[0]
initial_variants = variants_for(st.session_state.make, st.session_state.car_model)
if not initial_variants:
    st.error("No trained variant choices were found for this model.")
    st.stop()
if "variant" not in st.session_state or st.session_state.variant not in initial_variants:
    st.session_state.variant = "vxi" if "vxi" in initial_variants else initial_variants[0]


st.markdown(
    """
    <div class="hero">
      <p class="hero-kicker">Roadwise · Project prototype</p>
      <p class="hero-title">A clearer starting point<br>for your next car price.</p>
      <p class="hero-copy">Describe the vehicle and its listing details. The trained model will return an estimated sale price in the same units as the project data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.22, 0.78], gap="large")

with left:
    st.markdown('<p class="section-label">Vehicle profile</p><p class="section-title">Tell us about the car</p>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            make = st.selectbox("Make", make_options, key="make", on_change=change_make)
            fuel_type = st.selectbox("Fuel type", category_options["fuel_type"], index=preferred(category_options["fuel_type"], "petrol"), key="fuel_type")
            transmission = st.selectbox("Transmission", category_options["transmission"], index=preferred(category_options["transmission"], "manual"), key="transmission")
        with c2:
            current_models = models_for(make)
            if not current_models:
                st.error("No compatible models are available for this make.")
                st.stop()
            car_model = st.selectbox("Model", current_models, key="car_model", on_change=change_model)
            body_type = st.selectbox("Body type", category_options["body_type"], index=preferred(category_options["body_type"], "hatchback"), key="body_type")
            kms_run = st.number_input("Kilometres driven", min_value=0, max_value=996609, value=40000, step=1000, help="The largest mileage in the supplied dataset is 996,609 km.", key="kms_run")
        with c3:
            current_variants = variants_for(make, car_model)
            if not current_variants:
                st.error("No compatible variants are available for this make and model.")
                st.stop()
            variant = st.selectbox("Variant", current_variants, key="variant")
            total_owners = st.selectbox("Number of owners", list(range(1, 7)), index=0, key="total_owners")
            year_made = st.number_input("Manufacturing year", min_value=1996, max_value=2021, value=2018, step=1, key="year_made")

    st.markdown('<p class="section-label" style="margin-top:1.25rem">Listing details</p><p class="section-title">Where and when was it listed?</p>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            city = st.selectbox("Listing city", category_options["city"], index=preferred(category_options["city"], "new delhi"), key="city")
            registered_city = st.selectbox("Registered city", category_options["registered_city"], index=preferred(category_options["registered_city"], "delhi"), key="registered_city")
        with c2:
            registered_state = st.selectbox("Registered state", category_options["registered_state"], index=preferred(category_options["registered_state"], "delhi"), key="registered_state")
            rto = st.selectbox("RTO code", category_options["rto"], index=preferred(category_options["rto"], "dl1c"), key="rto")
        with c3:
            source = st.selectbox("Listing source", category_options["source"], index=preferred(category_options["source"], "online"), key="source")
            listing_date = st.date_input("Listing date", value=DATE_MAX, min_value=DATE_MIN, max_value=DATE_MAX, help="The training data only covers listings from 20 Feb 2019 to 30 May 2021.", key="listing_date")

    st.markdown('<p class="section-label" style="margin-top:1.25rem">Additional features</p><p class="section-title">Marketplace and ownership details</p>', unsafe_allow_html=True)
    with st.container(border=True):
        b1, b2, b3 = st.columns(3)
        with b1:
            assured_buy = st.checkbox("Assured buy", value=False, key="assured_buy")
        with b2:
            fitness_certificate = st.checkbox("Fitness certificate", value=False, key="fitness_certificate")
        with b3:
            warranty_avail = st.checkbox("Warranty available", value=False, key="warranty_avail")
        st.caption("These listing attributes were used during training, so include them even when the answer is ‘No’.")

    car_age = listing_date.year - int(year_made)
    input_row = {
        "fuel_type": fuel_type,
        "kms_run": int(kms_run),
        "city": city,
        "body_type": body_type,
        "transmission": transmission,
        "variant": variant,
        "assured_buy": bool(assured_buy),
        "registered_city": registered_city,
        "registered_state": registered_state,
        "rto": rto,
        "source": source,
        "make": make,
        "model": car_model,
        "total_owners": int(total_owners),
        "fitness_certificate": bool(fitness_certificate),
        "warranty_avail": bool(warranty_avail),
        "ad_year": listing_date.year,
        "ad_month": listing_date.month,
        "ad_dayofweek": listing_date.weekday(),
        "car_age": car_age,
    }
    input_signature = signature_for(input_row)

    estimate_clicked = st.button("Estimate this car's price", type="primary", use_container_width=True)
    if estimate_clicked:
        if car_age < 0:
            st.session_state.pop("latest_estimate", None)
            st.error("The manufacturing year cannot be later than the listing year.")
        elif car_age > 50:
            st.session_state.pop("latest_estimate", None)
            st.error("Please check the manufacturing year; this model only supports car ages up to 50 years.")
        else:
            try:
                model_input = pd.DataFrame([input_row], columns=feature_names)
                predicted_price = float(model.predict(model_input)[0])
                if predicted_price < 0:
                    st.session_state.pop("latest_estimate", None)
                    st.error("The model returned a negative estimate for these details. Try a different supported combination.")
                else:
                    st.session_state.latest_estimate = {
                        "signature": input_signature,
                        "price": predicted_price,
                        "input": input_row,
                        "created_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                    }
            except Exception as exc:
                st.session_state.pop("latest_estimate", None)
                st.error("The model could not calculate this estimate. Please review the selected details and try again.")
                st.caption(str(exc))

with right:
    estimate = st.session_state.get("latest_estimate")
    is_current = estimate is not None and estimate.get("signature") == input_signature

    with st.container(border=True):
        st.markdown('<p class="result-label">Your estimate</p>', unsafe_allow_html=True)
        if is_current:
            price = estimate["price"]
            st.markdown(f'<div class="price-value">₹ {price:,.0f}</div>', unsafe_allow_html=True)
            st.markdown('<span class="result-tag">Model estimate</span>', unsafe_allow_html=True)
            st.write("")
            st.caption("A point estimate from the trained project model. It is not a guaranteed sale price or a confidence interval.")
            st.divider()
            st.markdown(f'**{make.title()} {car_model.title()}** · {variant.title()}')
            st.caption(f"{kms_run:,.0f} km · {car_age} years old · {city.title()}")
            download_row = {**estimate["input"], "estimated_sale_price": price}
            download_data = pd.DataFrame([download_row]).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download this estimate",
                data=download_data,
                file_name="used_car_estimate.csv",
                mime="text/csv",
                use_container_width=True,
            )
        elif estimate is not None:
            st.markdown("### Inputs changed")
            st.write("Click **Estimate this car's price** to calculate an updated result for the current details.")
        else:
            st.markdown("### Ready when you are")
            st.write("Complete the vehicle details, then run the estimator to see the model output here.")
            st.markdown("<span class='result-tag'>No database or account needed</span>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<p class="section-label">About this model</p><p class="section-title">What powers the estimate</p>', unsafe_allow_html=True)
        st.metric("Algorithm", "Gradient Boosting")
        st.metric("Trees in the saved model", "300")
        st.metric("Cleaned source rows", "7,396 listings")
        st.caption("The supplied dataset spans Feb 2019–May 2021. The saved model has not been recalibrated for today's market.")

st.markdown("---")
st.markdown(
    '<p class="microcopy">Roadwise is a project demonstration. Estimates depend on the quality and age of the source data.</p>',
    unsafe_allow_html=True,
)
