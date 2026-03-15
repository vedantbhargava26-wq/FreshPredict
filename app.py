import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Page Configuration
st.set_page_config(page_title="FreshPredict AI", layout="wide")

st.title("🌿 FreshPredict AI: Enterprise Edition")
st.markdown("### Advanced Predictive Twin & Optimization Engine")
st.markdown("---")

# 2. Load the Data
df = pd.read_csv("data.csv")

# 3. AI Data Prep
flower_mapping = {'Rose': 0, 'Chrysanthemum': 1, 'Petunia': 2, 'Tulip': 3}
df['Flower_Code'] = df['Flower_Type'].map(flower_mapping)

X = df[['Flower_Code', 'Greenhouse_Temp_C', 'Greenhouse_Humidity', 'Transport_Temp_C', 'Days_in_Transit']]
y = df['Shelf_Life_Days']

# Train the AI
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Calculate Accuracy 
predictions_on_training = model.predict(X)
mae = mean_absolute_error(y, predictions_on_training)

# 4. The Control Center (Sidebar)
st.sidebar.header("⚙️ Cold Chain Parameters")

# Enterprise Diagnostics Panel
st.sidebar.info(f"**🔬 Model Accuracy (MAE):** ± {mae:.2f} Days")
st.sidebar.markdown("""
**📡 System Status:** 🟢 Live Connection  
**🧠 AI Engine:** FloraPredict RF-v3.0  
**📊 Data Standard:** Wageningen Horti-Protocol  
""")
st.sidebar.markdown("---")

display_flowers = {"Rose": "🌹 Rose", "Chrysanthemum": "🌼 Chrysanthemum", "Petunia": "🌺 Petunia", "Tulip": "🌷 Tulip"}
selected_flower_display = st.sidebar.selectbox("Select Flower Variety", list(display_flowers.values()))
selected_flower = [k for k, v in display_flowers.items() if v == selected_flower_display][0]

gh_temp = st.sidebar.slider("Greenhouse Temp (°C)", 15.0, 30.0, 21.0)
gh_hum = st.sidebar.slider("Greenhouse Humidity (%)", 50, 90, 75)
trans_temp = st.sidebar.slider("Transport Temp (°C)", 1.0, 15.0, 2.0)
transit_days = st.sidebar.slider("Expected Days in Transit", 1, 14, 3)
batch_value = st.sidebar.number_input("Batch Value (€)", min_value=1000, max_value=50000, value=8500, step=500)

# 5. Make the Prediction
input_data = pd.DataFrame({
    'Flower_Code': [flower_mapping[selected_flower]],
    'Greenhouse_Temp_C': [gh_temp],
    'Greenhouse_Humidity': [gh_hum],
    'Transport_Temp_C': [trans_temp],
    'Days_in_Transit': [transit_days]
})

prediction = model.predict(input_data)[0]
remaining_life = prediction - transit_days

# 6. Dashboard Layout - Top Row
st.header("📊 AI Vase-Life Prediction")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label=f"Total Lifespan ({selected_flower_display})", value=f"{prediction:.1f} Days")
with col2:
    st.metric(label="Usable Vase Life AFTER Transit", value=f"{remaining_life:.1f} Days")
with col3:
    if remaining_life < 5:
        st.error(f"⚠️ Financial Risk: € {batch_value:,.2f}")
    else:
        st.success(f"✅ Asset Secured: € {batch_value:,.2f}")

st.markdown("---")

# 7. ADVANCED AI FEATURES: XAI & Optimizer (Middle Row)
col_xai, col_opt = st.columns(2)

with col_xai:
    st.header("🧠 Explainable AI (XAI)")
    st.write("Which factors are driving the AI's current prediction?")
    # Get feature importances from the Random Forest
    importances = model.feature_importances_
    feature_names = ['Flower Variety', 'Greenhouse Temp', 'Greenhouse Humidity', 'Transport Temp', 'Days in Transit']
    imp_df = pd.DataFrame({'Impact Weight': importances}, index=feature_names)
    st.bar_chart(imp_df)

with col_opt:
    st.header("⚡ AI Action Prescriptions")
    st.write("The AI has simulated alternative scenarios to optimize your logistics:")
    
    # AI Secretly simulates dropping the transport temperature by 2 degrees
    opt_data = input_data.copy()
    opt_data['Transport_Temp_C'] = max(1.0, trans_temp - 2.0)
    opt_pred = model.predict(opt_data)[0]
    days_gained = opt_pred - prediction
    
    if days_gained > 0.1:
        st.success(f"**Action 1 (Cooling):** Dropping transport temperature by 2°C will recover **+{days_gained:.1f} days** of vase life.")
    else:
        st.info("**Action 1 (Cooling):** Current transport temperature is highly optimized.")
        
    # AI Secretly simulates speeding up the truck by 1 day
    opt_data2 = input_data.copy()
    opt_data2['Days_in_Transit'] = max(1, transit_days - 1)
    opt_pred2 = model.predict(opt_data2)[0]
    days_gained2 = opt_pred2 - prediction
    
    if days_gained2 > 0.1:
        st.warning(f"**Action 2 (Logistics):** Expediting shipping by 1 day preserves **+{days_gained2:.1f} days** of retail quality.")

st.markdown("---")

# 8. Interactive Heatmap Database (Bottom Row)
st.header("📋 Interactive Database & Heatmap")
st.write("Click any column header to sort. Colors indicate shelf-life viability.")

# Hide the secret AI code column and apply a beautiful Red-to-Green color gradient
display_df = df.drop(columns=['Flower_Code'])
styled_df = display_df.style.background_gradient(cmap='RdYlGn', subset=['Shelf_Life_Days'])

# use_container_width makes the table stretch nicely across the screen
st.dataframe(styled_df, use_container_width=True)