# Complaint Resolution Copilot

[Live Streamlit demo](https://complaint-resolution-copilot.streamlit.app/) · Portfolio prototype

This app helps a human reviewer triage **fictional** customer complaints. It predicts a service category, retrieves two relevant **mock** policies, lets the reviewer select one or both, and prepares a fixed policy-based draft. It does not send messages, verify customers, access bank systems, or provide actual bank policy.

## How it works

1. A complaint enters a TF-IDF (word unigrams and bigrams) + logistic regression classifier.
2. The classifier proposes one of three routes: **bank account**, **credit card**, or **mortgage**.
3. A separate TF-IDF search ranks mock policies within the predicted category and displays the top two.
4. A reviewer selects the applicable policy or policies. The app shows a fixed draft for human review. The CC01 + CC02 combination has a dedicated draft covering both card blocking and a disputed charge.

The similarity values shown in the app rank policy text matches. They are **not** probabilities that a policy is correct.

## Data and evaluation

The classifier used public CFPB complaint narratives from January–February 2025. After preparation, the data was split by time into **2,079 training**, **352 validation**, and **326 later-period test** complaints. The only model input was complaint text. Labels were grouped into the three service categories above.

| Final test metric | Most-frequent-category baseline | Text classifier |
| --- | ---: | ---: |
| Macro-F1 | 0.191 | **0.847** |
| Accuracy | — | **87.1%** |

On the same 326 test cases, the text model alone was correct on 161 cases and the baseline alone on 8 (paired exact test p ≈ 3.92 × 10⁻³⁸). A separate **nine-case, author-written** policy-retrieval check found the expected policy at rank 1 in 8/9 cases and in the top 2 in 9/9. The small retrieval check is illustrative, not an estimate of production performance.

## What testing revealed

| Complaint scenario | Observation | Design response |
| --- | --- | --- |
| Bank login problem | Correct route; BA02 access policy ranked second behind BA01 transaction policy. | Show two policies and require the reviewer to choose BA02. |
| Stolen credit card plus disputed charge | Both CC01 and CC02 appeared in the top two; a single-policy draft covered only one issue. | Allow both policies to be selected and use a combined, reviewed draft. |
| Mortgage statement balance | Correct route and top-ranked MT02. Selecting MT03 instead produced an irrelevant hardship draft. | Require an active policy choice; the reviewer must check that the selection fits the complaint. |

The mortgage example demonstrates that human selection can also be wrong. The app does not guarantee a correct draft, and no draft should be sent without substantive review.

## Why the draft is fixed

A small language model was tried during development. In one test it produced repetitive text; in another it instructed the customer to provide credentials. That output was rejected. The deployed app uses reviewed fictional templates instead of live language-model generation. Do not describe this deployment as a live LLM or agent.

## Run the repository

Keep `app.py`, `complaint_router.joblib`, `mock_policies.csv`, and `requirements.txt` in the repository root. The model was saved with **scikit-learn 1.6.1**; use a compatible Python environment, preferably **Python 3.13**, with the versions in `requirements.txt`.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Only load the `.joblib` model supplied with this project; deserializing an untrusted model file can execute code. The app is deployed through Streamlit Community Cloud from `main` and `app.py`.

## Scope and limitations

- CFPB narratives are public **consumer allegations**, not verified bank records or data from IDFC FIRST Bank or any other employer.
- Mock policy text is fictional and cannot be treated as actual financial-service guidance.
- The classifier handles only three categories and proposes one route, even if a complaint spans categories.
- Policy retrieval is lexical, and its nine-case check was small and author-written.
- Reviewers can select an irrelevant policy. There is no customer authentication, case-management integration, live resolution, or automatic sending.

**Interview summary:** “I built and evaluated a text classifier for complaint routing, added policy retrieval and reviewer-selected drafting, and used failure cases to improve the workflow. The later-period classifier test reached 0.847 macro-F1 on 326 complaints. The prototype deliberately requires a human reviewer and uses fictional policies.”
