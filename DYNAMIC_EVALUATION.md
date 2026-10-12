# Dynamic evaluation in LeadIQ

This update preserves the nine-step workflow and its existing model-training, calibration, scoring, and SHAP methodology. It does not use example outcomes or replace successful calculations.

## Step 6

Ten tabs are available: Overview, Model Comparison, ROC Curve, Precision-Recall Curve, Confusion Matrix, Learning Curve, Lift & Gain, Calibration, Feature Importance, and Verification.

The shared model selector changes **inspection only**. The training-CV winner still powers Steps 7 and 8. Overview reports accuracy, precision, recall, F1, ROC-AUC, average precision (PR-AUC), and precision at the top 10%, along with dataset, target, positive class, actual split counts, calibration, saved threshold, and result fingerprints. Confidence intervals belong only to the winner's stored predictions and operating point; they are not copied to other models.

Each candidate's lift, gain, reliability, and threshold calculations use its own saved prediction arrays. Candidate learning curves and actual SHAP/permutation analyses are calculated on demand, cached by training-run/model identity, and show loading or precise failure states. The Dummy baseline has no supported SHAP explanation; computed permutation zeros are shown as zero, not treated as missing. Learning curves concern the underlying uncalibrated candidate and training-only CV, not a held-out tuning procedure. SHAP output scope remains explicit.

The lift axis adapts above 1; a reference at 1 is explicitly labeled as no-skill lift. Probability axes remain bounded to 0–1. Constant-probability reliability is shown as an actual single point. Feature names are abbreviated on axes only; full names remain in tooltips and detail tables.

## Threshold exploration

Selecting a candidate restores that candidate's saved threshold. Changing the slider recomputes TN, FP, FN, TP, accuracy, precision, recall and F1. Aborted requests and a request sequence guard prevent late responses from replacing newer selections. Counts are hidden during recalculation and checked against model/run/threshold identity before rendering. Undefined precision with no predicted positives is explicitly described; its numeric zero follows sklearn's `zero_division=0` convention.

The explorer never saves a new operating point or changes probabilities, probability-ordering metrics, scores, or contact order. The response includes the saved prediction fingerprint so exploratory results are traceable.

## Freshness and provenance

Confirmed selection changes invalidate downstream results. Beginning cleaning, feature creation, training, or scoring revokes dependent completion immediately, removes dependent artifacts, and revokes previously issued download links before background processing starts. Failure does not restore old output. New training runs have a unique ID, timestamp, dataset fingerprint, target mapping, and held-out row fingerprint. Existing successful runs receive a deterministic retained-run identity without modifying their numerical results.

The browser refreshes project revision state while the workspace is active and on focus. Changed revisions remount dependent views; direct navigation to unavailable steps shows an invalidated state, not cached charts. Unsaved target/role/inclusion changes block downstream views until confirmation. Read APIs reject unavailable or mismatched runs. Model-analysis jobs use the project processing lock so replacement processing cannot interleave with analysis saves.

Verification recomputes all seven metrics and the confusion matrix from each requested model's saved probabilities, labels and saved threshold, then compares them with the stored displayed values. It displays actual matches or discrepancies; it does not automatically correct discrepancies.

## API additions

- `GET /api/projects/{pid}/evaluation?model=...&run_id=...`
- `POST /api/projects/{pid}/evaluation-analysis` with model and run_id
- Existing threshold and verification APIs now support run identity and model-specific evidence.

Every project endpoint still requires the project's existing `X-Project-Key`. APIs, datasets, trained artifacts, and job stages remain real. No hosted LLM service or additional credentials are required.

## Assumptions and limitations

- PR-AUC means average precision, not a trapezoidal PR integration.
- Candidate inspection cannot replace the scoring winner; doing so would change the approved methodology.
- Only the winner was calibrated during training. Uncalibrated candidates do not acquire an invented after-calibration curve.
- On-demand analyses can take time on large datasets. Failed or unsupported analyses leave unrelated metrics usable and give a retry or explicit reason.
- Global SHAP and permutation sampling limits are disclosed. They remain explanations of the underlying estimator, not causation or percentile-score contributions.
- Existing historical runs without a training timestamp say that it was not recorded. No timestamp is fabricated.
- Separately retained run history and future-outcome monitoring are not part of this targeted update.