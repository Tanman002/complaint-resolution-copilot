# Complaint Resolution Copilot · Streamlit preview

This portfolio prototype predicts which of three teams should review a fictional complaint, retrieves two relevant fictional policies, and displays a policy-specific fixed draft for human review. It does not send messages or access real bank systems. A small language model was tested separately and rejected after it instructed a customer to share credentials.

## Add your Colab assets

Unzip `complaint_copilot_assets.zip` from Colab. Place its two files, `complaint_router.joblib` and `mock_policies.csv`, alongside `app.py`. Never load a `.joblib` file from an untrusted source.

## Deploy

Create a GitHub repository; upload `app.py`, `requirements.txt`, and your two asset files at the repository root. In Streamlit Community Cloud choose that repository, branch, and `app.py` as the entrypoint. The requirements pin scikit-learn 1.6.1, matching the version used to save this model in Colab.

## Evidence and limits

The classifier was trained on 2,079 historical CFPB narratives from January–February 2025. A validation period had 352 rows; the final later-period test had 326 rows and produced 87.1% accuracy and 0.847 macro-F1, versus baseline macro-F1 0.191. An exact paired test on the same test examples counted 161 wins for the model alone versus 8 for the baseline alone (p ≈ 3.92 × 10⁻³⁸). A nine-case authored policy-retrieval exercise yielded 8/9 top-1 and 9/9 top-2 matches; it is a small demonstration and cannot establish real-world retrieval quality. CFPB narratives are allegations, not verified events, and the fictional bank policies are not actual bank guidance. Human review is required.
