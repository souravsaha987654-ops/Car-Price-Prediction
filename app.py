import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor

# 1. Page Configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="pngfind.com-car-gif-png-4213374.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .header-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    }
    .header-banner h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff;
    }
    .header-banner p {
        margin: 6px 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    .card {
        background-color: #ffffff;
        padding: 20px 24px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%) !important;
        color: white !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(255, 75, 43, 0.35) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 75, 43, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Model and Dataset
@st.cache_resource
def load_model():
    try:
        model = XGBRegressor()
        model.load_model("model.json")
        return model
    except Exception:
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Cardetails.csv")
        df['name'] = df['name'].apply(lambda x: str(x).split(' ')[0].strip())
        return df
    except Exception:
        return pd.DataFrame()

model = load_model()
df = load_data()

# 4. Error State Handling
if model is None:
    st.error("⚠️ **Model file `model.json` not found!**")
    st.info("Make sure you exported your trained XGBoost model from Google Colab as `model.json` and saved it inside this project folder.")
    st.stop()

# 5. Sidebar Inputs (Configured for your 11 Model Features)
st.sidebar.markdown("## ⚙️ Vehicle Parameters")

st.sidebar.subheader("General Specs")
brand_options = sorted(df["name"].unique()) if not df.empty else ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Mahindra', 'Tata']
name = st.sidebar.selectbox("Car Brand", brand_options)
year = st.sidebar.slider("Manufacturing Year", 1995, 2026, 2017)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=45000, step=2500)
seats = st.sidebar.slider("Seating Capacity", 2, 10, 5)

st.sidebar.subheader("Engine & Performance")
mileage = st.sidebar.slider("Mileage (kmpl)", 5.0, 45.0, 19.0, step=0.5)
engine = st.sidebar.slider("Engine Capacity (CC)", 500, 5000, 1197, step=50)
max_power = st.sidebar.slider("Max Power (bhp)", 20.0, 500.0, 82.0, step=1.0)
fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])

st.sidebar.subheader("Sales & History")
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
seller_type = st.sidebar.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
owner = st.sidebar.selectbox("Owner History", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Calculate Estimated Price", use_container_width=True)

# 6. Header Banner
st.markdown("""
    <div class="header-banner">
        <h1>Used Car Valuation ML model</h1>
        <p>Real-time market valuation powered by machine learning algorithms</p>
    </div>
""", unsafe_allow_html=True)

# 7. Main Dashboard Screen
current_year = 2026
car_age = max(0, current_year - year)

if predict_btn:
    # Prepare Input DataFrame
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    # Encodings matching model training
    owner_map = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5}
    fuel_map = {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4}
    seller_map = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
    trans_map = {'Manual': 1, 'Automatic': 2}
    
    brand_list = sorted(df['name'].unique()) if not df.empty else brand_options
    brand_map = {brand: idx + 1 for idx, brand in enumerate(brand_list)}

    input_data_model['owner'] = input_data_model['owner'].map(owner_map).fillna(1)
    input_data_model['fuel'] = input_data_model['fuel'].map(fuel_map).fillna(1)
    input_data_model['seller_type'] = input_data_model['seller_type'].map(seller_map).fillna(1)
    input_data_model['transmission'] = input_data_model['transmission'].map(trans_map).fillna(1)
    input_data_model['name'] = input_data_model['name'].map(brand_map).fillna(1)

    try:
        # Prediction
        predicted_price = float(model.predict(input_data_model)[0])
        predicted_price = max(10000.0, predicted_price)  # Floor value safeguard

        lower_bound = predicted_price * 0.92
        upper_bound = predicted_price * 1.08
        price_lakhs = predicted_price / 100000.0

        st.subheader("📊 Price Estimation Results")

        # Top 3 Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated Valuation", f"₹ {predicted_price:,.0f}", f"{price_lakhs:.2f} Lakhs")
        m2.metric("Expected Price Range", f"₹ {lower_bound:,.0f} - {upper_bound:,.0f}")
        m3.metric("Vehicle Age", f"{car_age} Years", delta="-Depreciation" if car_age > 3 else "High Resale")

        st.markdown("---")

        # Gauge Chart + Price Factors
        col_factors, col_gauge = st.columns([1.2, 1], gap="large")

        with col_factors:
            st.markdown('<div class="card"><div class="card-title">🔍 Market Valuation Factors</div>', unsafe_allow_html=True)
            
            factors = []
            if car_age <= 3:
                factors.append("🟢 **Low Age:** Minimal depreciation; retains prime market demand.")
            elif car_age <= 7:
                factors.append("🟡 **Moderate Age:** Average market depreciation rate.")
            else:
                factors.append("🔴 **High Age:** Significant depreciation impact on valuation.")

            if km_driven < 40000:
                factors.append("🟢 **Low Mileage:** Clean vehicle usage adds positive resale equity.")
            elif km_driven < 90000:
                factors.append("🟡 **Moderate Usage:** Typical wear & tear range.")
            else:
                factors.append("🔴 **High Mileage:** Higher mechanical wear penalizes valuation.")

            if transmission == "Automatic":
                factors.append("✨ **Transmission:** Automatic variants command a price premium.")
            if max_power > 120:
                factors.append("⚡ **Engine Power:** High power output (>120 bhp) improves pricing.")
            if owner == "First Owner":
                factors.append("👤 **Single Ownership:** High buyer preference and quicker saleability.")

            for f in factors:
                st.markdown(f"- {f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_gauge:
            st.markdown('<div class="card"><div class="card-title">🧭 Valuation Indicator</div>', unsafe_allow_html=True)
            
            gauge_max = max(15.0, price_lakhs * 1.5)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=price_lakhs,
                number={'prefix': "₹ ", 'suffix': " L"},
                gauge={
                    'axis': {'range': [0, gauge_max], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': "#2563eb"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#cbd5e1",
                    'steps': [
                        {'range': [0, gauge_max * 0.35], 'color': "#fee2e2"},
                        {'range': [gauge_max * 0.35, gauge_max * 0.70], 'color': "#fef3c7"},
                        {'range': [gauge_max * 0.70, gauge_max], 'color': "#dcfce7"}
                    ],
                    'threshold': {
                        'line': {'color': "#dc2626", 'width': 4},
                        'thickness': 0.8,
                        'value': price_lakhs
                    }
                }
            ))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Car Details Summary Card
        st.markdown('<div class="card"><div class="card-title">📋 Configured Vehicle Summary</div>', unsafe_allow_html=True)
        sum1, sum2, sum3 = st.columns(3)
        with sum1:
            st.write(f"• **Brand & Model:** {name}")
            st.write(f"• **Model Year:** {year} ({car_age} yrs old)")
            st.write(f"• **Fuel Type:** {fuel}")
            st.write(f"• **Seating:** {seats} Seater")
        with sum2:
            st.write(f"• **Kilometers:** {km_driven:,} km")
            st.write(f"• **Transmission:** {transmission}")
            st.write(f"• **Mileage:** {mileage} kmpl")
        with sum3:
            st.write(f"• **Engine CC:** {engine} cc")
            st.write(f"• **Max Power:** {max_power} bhp")
            st.write(f"• **Ownership:** {owner}")
            st.write(f"• **Seller Type:** {seller_type}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Selling Recommendations
        st.markdown('<div class="card"><div class="card-title">💡 Actionable Tips to Maximize Selling Price</div>', unsafe_allow_html=True)
        st.markdown("""
        * **Complete Service Records:** Having stamped periodic maintenance history increases valuation by up to **5–10%**.
        * **Professional Detailing:** Interior dry-cleaning and paint correction make a strong first impression on buyers.
        * **Tire & Battery Inspection:** Good tread depth and a healthy battery warranty card build buyer confidence.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Pipeline Error: {e}")

else:
    # Default Welcome Screen
    st.info("👈 Set your car specifications in the sidebar and click **'Calculate Estimated Price'** to run the prediction.")

    st.markdown('<div class="card"><div class="card-title">📌 Quick Benchmark Estimates</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("**Compact City Hatchback**")
        st.caption("e.g., Maruti Swift / Hyundai i10")
        st.write("• Year: 2018 | 40,000 km")
        st.write("• Typical Range: **₹ 4.2L – ₹ 5.8L**")
    with b2:
        st.markdown("**Mid-Size Executive Sedan**")
        st.caption("e.g., Honda City / Hyundai Verna")
        st.write("• Year: 2017 | 60,000 km")
        st.write("• Typical Range: **₹ 5.5L – ₹ 7.5L**")
    with b3:
        st.markdown("**Compact / Mid SUV**")
        st.caption("e.g., Hyundai Creta / Tata Nexon")
        st.write("• Year: 2019 | 45,000 km")
        st.write("• Typical Range: **₹ 8.0L – ₹ 11.5L**")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🤖 Model Architecture</div>', unsafe_allow_html=True)
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Algorithm", "XGBoost Regressor")
    m_col2.metric("Trained Features", "11 Engineered Specs")
    m_col3.metric("Data Source", "Indian Auto Market Dataset")
    st.markdown('</div>', unsafe_allow_html=True)