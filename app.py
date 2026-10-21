"""Fictional complaint routing and policy review demo."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
LABELS = {"credit_card": "Credit card", "bank_account": "Bank account", "mortgage": "Mortgage"}

TEMPLATES = {
    "CC01": "Thank you for reporting that your card may be lost or stolen. Please use the bank's secure channel to request a card block. A card specialist can review the related transactions after account verification.",
    "CC02": "Thank you for reporting a credit card charge you do not recognize. Please report the transaction date, merchant and amount through a secure channel so a card specialist can review the dispute.",
    "CC03": "Thank you for raising a question about your credit card payment or fee. After secure account verification, a card specialist can review the payment posting date and fee details.",
    "BA01": "Thank you for reporting an account transaction you do not recognize. Please report it through the bank's secure channel. An authorized agent can verify the account and investigate the transaction.",
    "BA02": "Thank you for telling us that you are having trouble accessing your bank account. Please use the bank's secure account recovery process for help. For your security, do not share your password, PIN, or one-time passcode in a complaint message. A service agent can guide you through the next step after verification.",
    "BA03": "Thank you for raising a question about a deposit or transfer delay. After secure account verification, an account specialist can check its status and explain the next investigation step.",
    "MT01": "Thank you for raising a concern about your mortgage payment record. After secure verification, a servicing specialist can review the payment record and posting date.",
    "MT02": "Thank you for your question about your mortgage statement or balance. A servicing specialist can review the relevant record after secure account verification.",
    "MT03": "Thank you for telling us that you need mortgage payment assistance. A trained servicing specialist can discuss available options with you after secure verification.",
}

COMBINED_TEMPLATES = {
    frozenset({"CC01", "CC02"}): (
        "Thank you for telling us that your credit card was stolen and that you noticed a charge "
        "you do not recognize. Please use the bank's secure channel to request a card block. "
        "You can also report the transaction date, merchant and amount through a secure channel "
        "so a card specialist can review the dispute after account verification."
    ),
}


def draft_for(selected):
    """Use approved fictional wording for each reviewer-selected policy."""
    special = COMBINED_TEMPLATES.get(frozenset(selected))
    if special:
        return special
    return "\n\n".join(TEMPLATES[policy_id] for policy_id in selected)


@st.cache_resource
def load_assets():
    model = joblib.load(ROOT / "complaint_router.joblib")
    policies = pd.read_csv(ROOT / "mock_policies.csv")
    required = {"policy_id", "category", "topic", "guidance"}
    if not required.issubset(policies.columns):
        raise ValueError(f"Missing policy columns: {sorted(required - set(policies.columns))}")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    vectors = vectorizer.fit_transform(policies["topic"] + ". " + policies["guidance"])
    return model, policies, vectorizer, vectors


def retrieve(query, category, policies, vectorizer, vectors):
    eligible = policies[policies["category"] == category].copy()
    scores = cosine_similarity(vectorizer.transform([query]), vectors[eligible.index]).ravel()
    eligible["similarity"] = scores
    return eligible.sort_values("similarity", ascending=False).head(2)


st.set_page_config(page_title="Complaint Resolution Copilot", page_icon="📨")
st.title("Complaint Resolution Copilot")
st.caption("Portfolio prototype · fictional bank policies · human review required")
st.write("Enter a fictional complaint to see its proposed routing and service guidance.")

try:
    model, policies, searcher, policy_vectors = load_assets()
except (FileNotFoundError, ValueError, ImportError) as exc:
    st.error(f"App assets could not be loaded: {exc}")
    st.info("Place complaint_router.joblib and mock_policies.csv beside app.py.")
    st.stop()

example = "I cannot sign in to my bank account."
complaint = st.text_area("Fictional customer complaint", value=example, height=130)

if st.button("Analyze complaint", type="primary"):
    if len(complaint.strip()) < 12:
        st.warning("Enter a more complete fictional complaint.")
    else:
        category = str(model.predict([complaint])[0])
        retrieved = retrieve(complaint, category, policies, searcher, policy_vectors)
        st.session_state.analysis = {
            "complaint": complaint,
            "category": category,
            "matches": retrieved.to_dict("records"),
        }

if "analysis" in st.session_state:
    analysis = st.session_state.analysis
    st.subheader("Proposed route")
    st.write(LABELS.get(analysis["category"], analysis["category"]))
    st.subheader("Retrieved mock policies")
    choices = analysis["matches"]
    for item in choices:
        with st.expander(f'{item["policy_id"]} · {item["topic"]} · similarity {item["similarity"]:.2f}'):
            st.write(item["guidance"])

    if not choices or choices[0]["similarity"] == 0:
        st.warning("No wording match was found. Route to a human specialist.")
    else:
        st.write("A reviewer must select every policy that addresses the issue.")
        chosen = st.multiselect(
            "Policies selected for draft",
            [item["policy_id"] for item in choices],
            format_func=lambda pid: f'{pid} · {next(i["topic"] for i in choices if i["policy_id"] == pid)}',
        )
        if st.button("Show draft for review"):
            if not chosen:
                st.warning("Select at least one relevant policy before drafting.")
            else:
                st.subheader("Draft for human review")
                st.write(draft_for(chosen))
                st.caption(
                    f"Source: mock policies {', '.join(chosen)}. "
                    "Human review required. This app does not send messages or access bank accounts."
                )

st.divider()
st.caption("Classifier evaluation: 326 later-period test complaints · 87.1% accuracy · 0.847 macro-F1. Historical US public allegations are not verified bank records. Prototype policies are fictional.")
