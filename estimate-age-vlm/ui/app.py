import streamlit as st
from utils.api_client import estimate_deer_age

st.set_page_config(page_title="Deer Age Estimator", layout="centered")

st.title("Deer Age Estimator")
st.write("Upload a photo of a deer jaw/teeth to estimate its age.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width="stretch")    
if uploaded_file and st.button("Estimate Age"):
    with st.spinner("Analyzing..."):
        result = estimate_deer_age(uploaded_file)

    if result:
        final = result.get("final_classification", {})

        st.success(f"**Estimated Age:** {final.get('estimated_age', 'unknown')}")

        st.write("**Explanation:**")
        st.write(final.get("logic_path", "No explanation"))

        st.write("**Confidence:**")
        st.write(final.get("confidence_score", 0.0))
        cost = result.get("cost")
        if cost is not None:
            st.info(f"💸 Estimated Cost: ${cost:.6f}")
        # 👇 debug thêm nếu cần
        with st.expander("🔍 Detailed Analysis"):
            st.json(result.get("priority_analysis", {}))