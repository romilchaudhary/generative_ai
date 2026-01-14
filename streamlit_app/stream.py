import streamlit as st
import numpy as np
import pandas as pd

# stream.py

st.set_page_config(page_title="Two-column Streamlit App", layout="wide")

# Sidebar controls
st.sidebar.title("Controls")
dataset = st.sidebar.selectbox("Choose a dataset", ["Sine/Cosine", "Random"])
series = st.sidebar.multiselect("Select series to show (dropdown)", ["A", "B", "C"], default=["A", "B"])
n_points = st.sidebar.slider("Number of points", min_value=50, max_value=2000, value=500, step=50)
with st.sidebar.expander("Setup & Run", expanded=True):
    st.write("Quick steps to setup and run this Streamlit app:")
    st.markdown(
        "- Create & activate a virtual environment\n"
        "- Install dependencies\n"
        "- Run the app and open http://localhost:8501\n"
    )
    st.code("""# create & activate virtual environment
        python -m venv .venv
        # Windows
        .venv\\Scripts\\activate
        # macOS / Linux
        source .venv/bin/activate

        # install dependencies
        pip install streamlit pandas numpy

        # run the app from this folder
        streamlit run stream.py
        """, language="bash"
    )

# Generate data
x = np.linspace(0, 10, n_points)
df = pd.DataFrame(index=np.arange(n_points))
if dataset == "Sine/Cosine":
    df["A"] = np.sin(x)
    df["B"] = np.cos(x)
    df["C"] = np.sin(2 * x) * 0.5
else:  # Random
    rng = np.random.default_rng(42)
    df["A"] = rng.normal(0, 1, n_points).cumsum()
    df["B"] = rng.normal(0, 1, n_points).cumsum()
    df["C"] = rng.normal(0, 1, n_points).cumsum()

# Main layout with two columns
st.title("Example Streamlit App")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Chart")
    if series:
        st.line_chart(df[series])
    else:
        st.info("Select at least one series in the sidebar to display the chart.")

with col2:
    st.subheader("Data & Summary")
    if series:
        st.dataframe(df[series].head(100))
        stats = df[series].agg(["mean", "std", "min", "max"]).T
        st.table(stats)
    else:
        st.write("No series selected.")

# Footer / quick info
st.sidebar.markdown("---")
st.sidebar.write("Change controls to update the view.")