"""Streamlit app: upload one histopathology image, get a subtype and the reason."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from multiclass_classify import (  # noqa: E402
    BEST_SUBCLASSES,
    BUNDLE_PATH,
    DISPLAY_NAMES,
    EXPERIMENT_F1,
    SUMMARY_PATH,
    classify_image,
    load_bundle,
    load_plip,
)

st.set_page_config(page_title="Subtype classifier", layout="centered")


@st.cache_resource
def load_runtime():
    bundle = load_bundle(BUNDLE_PATH)
    processor, image_encoder, device = load_plip(bundle["plip_model"])
    return bundle, processor, image_encoder, device


def main() -> None:
    st.title("Breast histopathology subtype classifier")
    retrain_note = ""
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        ranked = sorted(summary["cv_f1"], key=summary["cv_f1"].get, reverse=True)[:3]
        if set(ranked) == set(BEST_SUBCLASSES):
            retrain_note = (
                f" Retraining the same head here reached {summary['cv_accuracy'] * 100:.2f}%, "
                "and the same three subtypes stayed on top."
            )
    st.write(
        "This page classifies one H&E region into the seven BRACS subtypes. "
        "It uses the setup that worked best in the completed experiments: "
        "a frozen PLIP embedding and a small 7-class adapter. "
        "That experiment reached 65.19% patient-isolated accuracy, ahead of "
        "cosine nearest-neighbor at 55.31%."
        + retrain_note
        + " The 8-class BreakHis subtype task stayed near 31–47%, so the app "
        "does not use that label set."
    )

    st.subheader("Three subtypes classified best")
    st.write(
        "Ranked by out-of-fold F1 of the PLIP adapter. "
        "Invasive carcinoma is also first under nearest-neighbor search. "
        "Ductal carcinoma in situ stays in the top three there as well. "
        "Flat epithelial atypia is second for the adapter and nearly tied "
        "for third under nearest-neighbor search."
    )
    columns = st.columns(3)
    for column, name in zip(columns, BEST_SUBCLASSES):
        column.metric(DISPLAY_NAMES[name], f"F1 {EXPERIMENT_F1[name]:.2f}")

    st.divider()
    uploaded = st.file_uploader(
        "Histopathology image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "webp"],
        help="Upload one breast histopathology region. The image is classified on this machine.",
    )
    if uploaded is None:
        st.info("Upload an image to classify it.")
        _disclaimer()
        return

    try:
        image = Image.open(uploaded)
        image.load()
    except Exception:
        st.error("That file could not be read as an image.")
        _disclaimer()
        return

    st.image(image, caption=uploaded.name, use_container_width=True)

    if not BUNDLE_PATH.exists():
        st.error(
            "The trained adapter is not on disk yet. "
            "From the project folder, run: python src/multiclass_classify.py --train"
        )
        _disclaimer()
        return

    with st.spinner("Classifying with PLIP and the subtype adapter..."):
        bundle, processor, image_encoder, device = load_runtime()
        result = classify_image(image, bundle, processor, image_encoder, device)

    st.subheader(result["display_name"])
    left, right = st.columns(2)
    left.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
    right.metric(
        "Next class",
        f"{result['runner_up_name']} ({result['runner_up_confidence'] * 100:.1f}%)",
    )
    if result["is_best_subclass"]:
        st.caption("This is one of the three subtypes the experiments separate most reliably.")
    else:
        st.caption("This subtype was harder to separate in the experiments than the three above.")

    st.markdown("**Reason**")
    st.write(result["reason"])

    scores = (
        pd.Series(result["probabilities"])
        .rename(index=DISPLAY_NAMES)
        .sort_values(ascending=False)
        .rename("probability")
    )
    st.markdown("**Class probabilities**")
    st.bar_chart(scores)
    st.dataframe(
        scores.map(lambda value: f"{value:.1%}").rename("probability").to_frame(),
        use_container_width=True,
        hide_index=False,
    )

    cv_accuracy = bundle.get("cv_accuracy")
    if cv_accuracy is not None:
        st.caption(
            f"This adapter's patient-isolated cross-validation accuracy is {cv_accuracy * 100:.2f}%. "
            "The deployed weights were then fit on all training regions, so that figure is the "
            "generalization estimate, not the score of the saved weights on images they have already seen."
        )
    _disclaimer()


def _disclaimer() -> None:
    st.divider()
    st.caption(
        "Research demonstration only. This is not a diagnostic device and must not be used "
        "to guide patient care. The class definitions in the reason describe the subtype name; "
        "they are not measurements taken from the uploaded image."
    )


main()
