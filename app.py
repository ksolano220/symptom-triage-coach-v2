import os

import streamlit as st

try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except FileNotFoundError:
    pass

from inference import triage  # noqa: E402

st.title("Symptom Triage Coach v2")
st.write("Upload a photo of the affected area and describe the symptom. Pre-visit prep, not medical advice.")

img = st.file_uploader("Photo", type=["png", "jpg", "jpeg", "webp"])
text = st.text_area("Describe the symptom")

if st.button("Run") and img and text.strip():
    with st.spinner("Calling model..."):
        result = triage(img.read(), text)

    st.subheader("Visual findings")
    for f in result["visual_findings"]:
        st.write(f"- {f}")

    st.subheader("Systems")
    st.write(", ".join(result["systems"]))

    st.subheader("Possible causes")
    for c in result["possible_causes"]:
        st.markdown(f"**{c['name']}** _{c['likelihood']}_  \n{c['description']}")

    if result["red_flags"]:
        st.subheader("Red flags")
        for r in result["red_flags"]:
            st.write(f"- {r}")

    st.subheader("Questions")
    for q in result["questions_to_prepare_for"]:
        st.write(f"- {q}")

    with st.expander("Raw JSON"):
        st.json(result)
