import os

import streamlit as st

st.set_page_config(page_title="Symptom Triage Coach v2", layout="centered")

try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except FileNotFoundError:
    pass

from inference import triage  # noqa: E402


LIKELIHOOD_COLOR = {
    "common": "blue",
    "less_common": "gray",
    "rare": "orange",
    "serious": "red",
}


st.title("Symptom Triage Coach v2")
st.caption("Pre-visit prep. Not medical advice.")

st.divider()

left, right = st.columns(2)
with left:
    img = st.file_uploader("Photo of the affected area", type=["png", "jpg", "jpeg", "webp"])
with right:
    text = st.text_area("Describe the symptom", height=158)

run = st.button("Run triage", type="primary", disabled=not (img and text.strip()))

if run:
    with st.spinner("Calling model..."):
        try:
            result = triage(img.read(), text)
        except Exception as e:
            st.error(str(e))
            st.stop()

    st.divider()

    findings_col, followup_col = st.columns(2)

    with findings_col:
        st.subheader("Visual findings")
        for f in result["visual_findings"]:
            st.write(f"- {f}")

        st.subheader("Body systems")
        st.write(", ".join(result["systems"]))

        st.subheader("Possible causes")
        for c in result["possible_causes"]:
            with st.container(border=True):
                color = LIKELIHOOD_COLOR.get(c["likelihood"], "gray")
                st.markdown(f"**{c['name']}** &nbsp; :{color}-badge[{c['likelihood']}]")
                st.write(c["description"])

    with followup_col:
        if result["red_flags"]:
            st.subheader("Red flags")
            st.error("\n\n".join(f"- {r}" for r in result["red_flags"]))

        st.subheader("Questions to prepare for")
        for q in result["questions_to_prepare_for"]:
            st.write(f"- {q}")

    with st.expander("Raw JSON"):
        st.json(result)
