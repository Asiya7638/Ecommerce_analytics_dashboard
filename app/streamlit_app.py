# app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import joblib

# 1) Must be first
st.set_page_config(
    page_title="E-Commerce Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Built by Asiya S • Data Science Portfolio"}
)

# 2) Load CLV model
artifact = joblib.load(Path(__file__).parent.parent / 'models' / 'clv_model.pkl')
clv_model, clv_snapshot = artifact['model'], artifact['snapshot']

# 3) CSS, navbar, background, hover
st.markdown("""
<style>
  body { background: linear-gradient(-45deg,#f5f7fa,#c3cfe2,#f5f7fa,#c3cfe2);
         background-size:400% 400%;
         animation:gradientBG 15s ease infinite; }
  @keyframes gradientBG {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
  }
  .stMetric > div:first-child:hover {
    background-color:#f0f8ff!important;
    cursor:pointer;
    transform:scale(1.02);
    transition:all .2s ease-in-out;
  }
  .nav-bar {
    position:fixed;top:0;left:0;right:0;height:3rem;
    background:rgba(255,255,255,0.9);
    display:flex;align-items:center;padding:0 2rem;
    box-shadow:0 2px 5px rgba(0,0,0,0.1);z-index:1000;
  }
  .nav-bar a {
    margin-right:1.5rem;text-decoration:none;color:#333;font-weight:500;
  }
  .nav-bar a:hover { color:#0b6cff; }
  .main > .block-container { padding-top:4rem; }
</style>
<div class="nav-bar">
  <a href="#🌐-Geo-&-Trends">Geo & Trends</a>
  <a href="#📦-Products">Products</a>
  <a href="#👤-RFM-Segments">RFM Segments</a>
  <a href="https://github.com/YourRepo" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)

# 4) Logo
logo = Path(__file__).parent / "assets" / "logos.png"
if logo.exists():
    st.image(str(logo), width=160)

# 5) Load data
@st.cache_data(show_spinner=False)
def load_data():
    root = Path(__file__).parent.parent
    df = pd.read_csv(
        root / 'data' / 'processed' / 'cleaned_retail.csv',
        parse_dates=['InvoiceDate']
    )
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    return df

df = load_data()

# 6) Sidebar
st.sidebar.header("Filters & Export")
min_d, max_d = df['InvoiceDate'].min(), df['InvoiceDate'].max()
start, end = st.sidebar.date_input(
    "Invoice Date Range", (min_d, max_d), min_value=min_d, max_value=max_d
)
countries = ['All'] + sorted(df['Country'].unique())
country = st.sidebar.selectbox("Country", countries)

mask = (df['InvoiceDate'] >= pd.to_datetime(start)) & (df['InvoiceDate'] <= pd.to_datetime(end))
if country != 'All':
    mask &= (df['Country'] == country)
data = df[mask]

snapshot = df['InvoiceDate'].max() + dt.timedelta(days=1)
rfm_sb = (
    df.assign(Revenue=df.Quantity * df.UnitPrice)
    .groupby('CustomerID')
    .agg(
        Recency=('InvoiceDate', lambda x: (snapshot - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('Revenue', 'sum')
    )
    .reset_index()
)
rfm_sb['RecencyScore'] = np.ceil(rfm_sb['Recency'].rank(pct=True, ascending=False) * 5).astype(int)
rfm_sb['FrequencyScore'] = np.ceil(rfm_sb['Frequency'].rank(pct=True, ascending=True) * 5).astype(int)
rfm_sb['MonetaryScore'] = np.ceil(rfm_sb['Monetary'].rank(pct=True, ascending=True) * 5).astype(int)
rfm_sb['RFM_Segment'] = (
    rfm_sb['RecencyScore'].astype(str) +
    rfm_sb['FrequencyScore'].astype(str) +
    rfm_sb['MonetaryScore'].astype(str)
)

segs = sorted(rfm_sb['RFM_Segment'].unique())
chosen = st.sidebar.multiselect("Filter by RFM Segment", segs)
if chosen:
    data = data.merge(rfm_sb[['CustomerID', 'RFM_Segment']], on='CustomerID', how='left')
    data = data[data['RFM_Segment'].isin(chosen)]

show_map = st.sidebar.checkbox("Show country map", True)
csv = data.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📥 Download filtered data", data=csv,
                           file_name="filtered_data.csv", mime="text/csv")

# 7) Title
st.title("📊 E-Commerce Analytics Dashboard")
st.markdown(f"**Showing:** {start:%b %d, %Y} → {end:%b %d, %Y} • **Country:** {country}")

# 8) KPIs
today, yesterday = end, end - dt.timedelta(days=1)
rev_t = data.loc[data['InvoiceDate'].dt.date == today, 'Revenue'].sum()
rev_y = data.loc[data['InvoiceDate'].dt.date == yesterday, 'Revenue'].sum()
cust_t = data.loc[data['InvoiceDate'].dt.date == today, 'CustomerID'].nunique()
cust_y = data.loc[data['InvoiceDate'].dt.date == yesterday, 'CustomerID'].nunique()
avg_o = data.groupby('InvoiceNo')['Revenue'].sum().mean()

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("🛒 Orders Today", f"{data['InvoiceNo'].nunique():,}")
with c2: st.metric("👥 Customers Today", f"{cust_t:,}", delta=f"{cust_t - cust_y:+,}")
with c3: st.metric("💷 Revenue Today", f"£{rev_t:,.2f}", delta=f"£{rev_t - rev_y:,.2f}")
with c4: st.metric("📈 Avg. Order Value", f"£{avg_o:,.2f}")

# 9) Sparkline
monthly = data.set_index('InvoiceDate').resample('M')['Revenue'].sum().reset_index()
with st.expander("📈 Monthly Revenue Sparkline"):
    fig_s = px.line(monthly, x='InvoiceDate', y='Revenue', height=120)
    fig_s.update_traces(line_color="#4CAF50",
      hovertemplate="Month: %{x|%b %Y}<br>Rev: £%{y:,.2f}<extra></extra>")
    fig_s.update_layout(margin=dict(l=0, r=0, t=0, b=0),
      xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(fig_s, use_container_width=True)

# 10) Tabs + CLV
tab1, tab2, tab3 = st.tabs(["🌐 Geo & Trends", "📦 Products", "👤 RFM Segments"])

with tab1:
    if show_map:
        st.subheader("📍 Revenue by Country")
        cr = data.groupby('Country')['Revenue'].sum().reset_index()
        fm = px.choropleth(cr, locations='Country', locationmode='country names',
            color='Revenue', color_continuous_scale=px.colors.sequential.Greens,
            hover_data={'Revenue': ':.2f'})
        fm.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fm, use_container_width=True)

    st.subheader("Monthly Revenue Trend")
    f1 = px.line(monthly, x='InvoiceDate', y='Revenue', markers=True,
                 labels={'InvoiceDate': 'Month', 'Revenue': '£ Revenue'})
    f1.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(f1, use_container_width=True)

    st.subheader("Revenue by Weekday")
    data['Weekday'] = data['InvoiceDate'].dt.day_name()
    wk = data.groupby('Weekday')['Revenue'].sum().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ).reset_index()
    f2 = px.bar(wk, x='Weekday', y='Revenue', labels={'Revenue': '£ Revenue'})
    st.plotly_chart(f2, use_container_width=True)

with tab2:
    st.subheader("Top Products by Quantity")
    top_n = st.slider("Show top N products:", 5, 20, 10)
    tp = (data.groupby('Description')
          .agg(Quantity=('Quantity', 'sum'),
               Orders=('InvoiceNo', 'nunique'),
               Revenue=('Revenue', 'sum'))
          .sort_values('Quantity', ascending=False)
          .head(top_n).reset_index())

    term = st.text_input("🔍 Search product description", "")
    if term:
        tp = tp[tp['Description'].str.contains(term, case=False, na=False)]

    thresh = tp['Revenue'].quantile(0.9)
    js = JsCode(f"""
      function(params) {{
        if(params.data.Revenue > {thresh:.2f}) {{
          return {{backgroundColor:'#e7f4e4'}};
        }}
      }};
    """)
    gb = GridOptionsBuilder.from_dataframe(tp)
    gb.configure_column("Revenue", cellStyle=js)
    gb.configure_default_column(sortable=True, filter=True, resizable=True)
    grid_opts = gb.build()
    AgGrid(tp, gridOptions=grid_opts, enable_enterprise_modules=False, allow_unsafe_jscode=True)

    st.subheader("Product Mix Treemap")
    f3 = px.treemap(tp, path=['Description'], values='Quantity',
                    color='Quantity', color_continuous_scale=px.colors.sequential.Greens)
    st.plotly_chart(f3, use_container_width=True)

with tab3:
    st.subheader("Customer RFM Segments")
    snap = data['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = (data.assign(Revenue=data.Quantity * data.UnitPrice)
           .groupby('CustomerID')
           .agg(Recency=('InvoiceDate', lambda x: (snap - x.max()).days),
                Frequency=('InvoiceNo', 'nunique'),
                Monetary=('Revenue', 'sum'))
           .reset_index())
    rfm['RecencyScore'] = np.ceil(rfm['Recency'].rank(pct=True, ascending=False) * 5).astype(int)
    rfm['FrequencyScore'] = np.ceil(rfm['Frequency'].rank(pct=True, ascending=True) * 5).astype(int)
    rfm['MonetaryScore'] = np.ceil(rfm['Monetary'].rank(pct=True, ascending=True) * 5).astype(int)
    rfm['RFM_Segment'] = (
        rfm['RecencyScore'].astype(str) +
        rfm['FrequencyScore'].astype(str) +
        rfm['MonetaryScore'].astype(str)
    )
    sc = (rfm['RFM_Segment']
          .value_counts()
          .sort_index()
          .rename_axis('Segment')
          .reset_index(name='Count'))
    f4 = go.Figure(go.Bar(x=sc['Segment'], y=sc['Count']))
    f4.update_layout(title="RFM Segment Counts",
                     xaxis_title="Segment", yaxis_title="Customers",
                     margin=dict(l=20, r=20, t=40, b=20), template="plotly_white")
    st.plotly_chart(f4, use_container_width=True)

    st.markdown("**Top 5 ‘555’ Customers by Spend**")
    t5 = rfm[rfm['RFM_Segment'] == '555'].nlargest(5, 'Monetary')[['Recency', 'Frequency', 'Monetary']]
    st.table(t5)

    # CLV Predictions
    st.subheader("🤝 Predicted 90-day CLV")
    cf = (data.assign(Revenue=data.Quantity * data.UnitPrice)
          .groupby('CustomerID')
          .agg(Recency=('InvoiceDate', lambda x: (clv_snapshot - x.max()).days),
               Frequency=('InvoiceNo', 'nunique'),
               Monetary=('Revenue', 'sum'))
          .reset_index())
    X_new = cf[['Recency', 'Frequency', 'Monetary']].to_numpy().astype(float)
    cf['CLV'] = clv_model.predict(X_new)
    top_clv = cf.nlargest(5, 'CLV')[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'CLV']]
    top_clv['CLV'] = top_clv['CLV'].map(lambda x: f"£{x:,.2f}")
    st.table(top_clv)

# Footer
st.markdown("<hr><center>Built by **Asiya S** • Data Science Portfolio</center>", unsafe_allow_html=True)







