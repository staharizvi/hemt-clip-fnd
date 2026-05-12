"""Streamlit demo for HEMT-CLIP.

Layout (Blueprint §11):
    - Text + image input + Analyze button
    - Prediction (Real/Fake) with confidence bars
    - CLIP alignment score alpha with verbal interpretation
    - Tabs: Attention Heatmap | SHAP Explanation | How it Works

Model loaded once via @st.cache_resource. Inference runs on CPU for
Streamlit Cloud, or via ngrok tunnel from Colab GPU during viva.
"""

# TODO: implement Streamlit UI
