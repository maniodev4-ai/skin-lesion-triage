"""
app/app.py - Streamlit front end for the skin-lesion triage prototype.
Run from the project root:   streamlit run app/app.py
EDUCATIONAL RESEARCH PROTOTYPE. NOT A MEDICAL DEVICE.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference as inf  # noqa: E402

LOCATIONS = ["unknown", "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", "hand",
             "lower extremity", "neck", "scalp", "trunk", "upper extremity"]

st.set_page_config(page_title="Skin-lesion triage (educational prototype)", layout="wide")
st.title("Skin-lesion triage: educational research prototype")
st.error("**NOT A MEDICAL DEVICE.** This is a student research prototype. It must not be used for real "
         "patient decisions, diagnosis, or to decide whether someone needs care. If you are worried about a "
         "skin lesion, see a doctor.")


@st.cache_resource(show_spinner="Loading the model...")
def load():
    return inf.load_system()


try:
    cfg, gate, members = load()
except Exception as e:  # missing model files, etc.
    st.error(f"The model could not be loaded: {e}")
    st.stop()

with st.sidebar:
    st.header("Optional patient details")
    st.number_input("Age in years (0 = unknown)", min_value=0, max_value=120, value=0)
    st.selectbox("Sex", ["unknown", "female", "male"])
    st.selectbox("Body location", LOCATIONS)
    st.info("You can leave these blank. The locked final model uses the **image only**: in testing, patient "
            "details gave no measurable gain, and hiding all of them changed nothing. Anything entered here is "
            "not used for the result.")
    st.header("About this model")
    st.caption(f"Ensemble of {len(members)} fine-tuned EfficientNet-B0 networks, running on CPU. "
               "Test set (1,431 images, 1,067 lesions): about 86% of high-concern lesions were sent to a "
               "doctor (95% CI 81 to 92%) at a 30% referral rate; for melanoma alone about 81%.")

uploaded = st.file_uploader("Upload a dermoscopic image (JPG or PNG)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.write("Upload an image to see the predicted lesion type, the confidence, the referral decision and a "
             "heatmap of where the model looked.")
    st.stop()

try:
    image = inf.prepare_image(uploaded.getvalue())
except inf.BadImage as e:
    st.error(f"Could not use this file. {e}")
    st.stop()
except Exception as e:  # anything unexpected must not crash the app
    st.error(f"Unexpected problem while reading the file: {e}")
    st.stop()

try:
    with st.spinner("Analysing..."):
        result = inf.analyse(image, cfg, gate, members)
except Exception as e:
    st.error(f"The analysis failed: {e}")
    st.stop()

decision = result["decision"]
left, right = st.columns([1, 1])
with left:
    st.subheader("Image and heatmap")
    a, b = st.columns(2)
    a.image(result["resized"], caption="Image as seen by the model (224x224)", use_container_width=True)
    b.image(result["overlay"], caption="Where the model looked (Grad-CAM)", use_container_width=True)
    st.caption("The heatmap shows where the model's attention was, not why it decided. It does not prove the "
               "model reasons like a dermatologist.")
with right:
    st.subheader("Decision")
    text = f"**{decision}**: {result['reason']}"
    if decision == "REFER":
        st.error(text)
    elif decision == "DEFER":
        st.warning(text)
    else:
        st.success(text)
    c1, c2, c3 = st.columns(3)
    c1.metric("Most likely type", result["pred"].upper(), inf.FULL_NAMES[result["pred"]])
    c2.metric("Confidence", f"{result['confidence']:.1%}")
    c3.metric("P(high-concern)", f"{result['p_high_concern']:.1%}")
    st.caption("High-concern = melanoma, basal cell carcinoma, actinic keratosis. Time taken: "
               f"{result['seconds']:.2f} s on CPU.")

st.subheader("Probability of each class")
table = pd.DataFrame({"class": [f"{k} ({inf.FULL_NAMES[k]})" for k in result["probs"]],
                      "probability": list(result["probs"].values())})
st.bar_chart(table.set_index("class")["probability"])
st.dataframe(table.assign(probability=table["probability"].map(lambda p: f"{p:.1%}")), hide_index=True)

with st.expander("Image quality check"):
    th = gate["thresholds"]
    q = result["quality"]
    st.table(pd.DataFrame({
        "check": ["sharpness", "noise", "brightness"],
        "this image": [f"{q['sharpness']:.1f}", f"{q['noise']:.2f}", f"{q['brightness']:.1f}"],
        "accepted range": [f">= {th['sharpness_min']:.1f}", f"<= {th['noise_max']:.2f}",
                           f"{th['brightness_min']:.1f} to {th['brightness_max']:.1f}"]}))
    st.write("Passed." if not result["quality_problems"] else "Problems: " + "; ".join(result["quality_problems"]))
    st.caption("Tests showed the model's accuracy collapses on blurred, noisy, dark or heavily compressed "
               "images, so blur, noise and brightness are checked first. Compression artifacts are NOT detected, "
               "and the check is imperfect.")
    detect = ", ".join(f"{k} {v:.0%}" for k, v in gate.get("detection_rates", {}).items())
    st.caption(f"Measured on held-out validation images: wrongly flags {gate.get('false_flag_rate_clean', 0):.1%} "
               f"of clean images; catches {detect}.")

with st.expander("How the decision rule works, and when not to use this"):
    st.markdown(
        f"- **REFER** if the estimated probability of melanoma, basal cell carcinoma or actinic keratosis is at "
        f"least {cfg['referral_threshold_p_hc']:.1%}. The cut-off was chosen so that at most about 30% of lesions "
        "are sent to a doctor.\n"
        f"- **DEFER** if the image fails the quality check, or the top class has under "
        f"{cfg['defer_confidence_tau']:.0%} confidence.\n"
        "- **NOT REFERRED** otherwise. This is never a clinical all-clear.\n"
        "- Known limits: about 1 in 5 high-concern lesions (about 1 in 5 melanomas) were still not sent in "
        "testing; trained on two sites (Vienna and Australia); performance falls on blurred, noisy, dark or "
        "heavily compressed images; not tested on other devices, skin types or populations.\n"
        "- Do not use it for real patients, for clinical triage, on non-dermoscopic photos (such as phone "
        "photos), or without a clinician.")