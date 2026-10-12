# LeadIQ — Targeted Dynamic Evaluation Update

LeadIQ turns an uploaded outcome-labeled dataset into evaluated predictions, ranked leads, and explainable results.
This update completes the dynamic evaluation experience without rebuilding the application, replacing working calculations, or changing its nine-step workflow.

## Who it's for

- Existing LeadIQ users evaluating their own datasets rather than a prescribed lead schema.
- Sales and operations teams needing consistent probabilities, rankings, and understandable explanations.
- Analysts and beginners comparing model performance while recognizing uncertainty and limitations.

## Core features and experience

### Preserve what is already implemented

Inspection of the current evaluation implementation shows that much of the requested functionality already exists:

- Actual model training and training-only cross-validation, a reserved evaluation split, winner selection, probability calibration, and an out-of-fold decision threshold.
- Computed accuracy, precision, recall, F1, ROC-AUC, PR-AUC, precision at the top 10%, confusion matrices, model comparisons, ROC and precision–recall curves.
- Actual positive-class prevalence, learning curves, cumulative gain, decile lift values, calibration results, SHAP and permutation importance for the selected winner.
- Recorded processing stages and live logs, saved test predictions, and a verification view.
- Model-based probability scoring, percentile lead scores, priority tiers, ranking, and individual SHAP explanations.

These capabilities are retained where they already satisfy the request. Source inspection establishes what is implemented, not a claim that every interaction has already been validated. No replacement models, illustrative results, or unnecessary redesign are proposed.

### Complete the evaluation workspace

Step 6 will contain these ten tabs, using the existing visual language:

1. **Overview:** the inspected model, all seven requested metrics, dataset and target context, split sizes, calibration status, relevant uncertainty, and warnings.
2. **Model Comparison:** all actual candidates, their measured results, comparison graph, selection criterion, and recorded operating thresholds.
3. **ROC Curve:** all applicable models, with their actual AUC values and an identified no-skill reference.
4. **Precision-Recall Curve:** all applicable models and the positive prevalence of the evaluated records.
5. **Confusion Matrix:** TP, TN, FP, FN and threshold-dependent metrics for the inspected model.
6. **Learning Curve:** measured results at 20%, 40%, 60%, 80%, and 100% of the training portion used within each validation fold, with actual row counts.
7. **Lift & Gain:** both an actual lift graph and cumulative gain graph, retaining the decile detail table.
8. **Calibration:** genuine reliability results, including before/after calibration when that comparison exists for the inspected model.
9. **Feature Importance:** actual SHAP and permutation importance for the inspected model where supported, with explanation scope and sampling limitations disclosed.
10. **Verification:** displayed values alongside independent recalculations from the corresponding saved predictions, with Match or discrepancy indicators.

The identified presentation gaps include the missing Overview tab and lift graph. Model-wide inspection and result-refresh behavior also need strengthening; the current model dropdown mainly controls the confusion matrix, while several other views remain tied to the winner.

### Make model selection consistent

- A shared evaluation model selector defaults to the cross-validation-selected winner.
- Changing the inspected model updates its overview metrics, confusion matrix, learning curve, lift/gain, and supported importance results. All-model comparison views retain their full set of candidates and identify the inspected model.
- Model-specific analyses not yet calculated show a calculation state followed by that model's genuine results, or a precise unavailable reason.
- Winner-only results are never relabeled as another model's results. Confidence intervals and calibration comparisons appear only for the model and predictions they actually describe.
- Calibration remains faithful to the current methodology: the winner has a before/after comparison; an uncalibrated candidate shows its actual reliability and a clear explanation that no after-calibration result exists for that run.
- Inspecting another model does **not** silently replace the scoring model. Steps 7 and 8 continue using the cross-validation-selected winner, clearly distinguished from the model being inspected in Step 6.

### Make threshold interaction consistent

- Selecting a model initializes the threshold from that model's saved operating point.
- Moving the threshold recalculates TP, TN, FP, FN, accuracy, precision, recall, and F1 from that model's saved probabilities and evaluation labels.
- The latest selected threshold wins if requests finish out of order. A recalculation indicator prevents old counts from appearing to describe a newly selected threshold.
- Probability-ordering metrics, ROC/PR curves, probabilities, lead scores, and rankings do not change merely because the classification threshold moves.
- Exploratory threshold changes remain separate from the saved scoring methodology and operating point; labels make that distinction visible.

### Prevent outdated or mismatched results

- Replacing a dataset or confirming changes to its target, positive class, included features, or other model-affecting settings invalidates dependent results.
- Unsaved selection changes cannot be presented as an evaluated configuration. After confirmation, fresh training is required before updated evaluation, rankings, explanations, or corresponding exports become available.
- Starting replacement processing marks dependent results unavailable immediately rather than leaving prior charts apparently current until the new run finishes.
- Failed or interrupted replacement processing does not restore old results as though they came from the new configuration.
- Completing a new run refreshes affected tables, charts, scores, and explanation availability. Refreshing the page or returning to an earlier tab does not reintroduce superseded results.
- Dataset identity, target and positive-class selection, training run, model, evaluation records, calibration status, and applicable threshold remain consistent across displayed results.

### Keep every result truthful and adaptive

- Every result series and metric comes from actual uploaded records and trained-model outputs. No hardcoded example scores, fabricated curves, random stand-ins, or fallback feature importance are introduced.
- Graph labels, feature names, available models, series, legends, and relevant axis ranges adapt to the results. Lift is not constrained to the 0–1 range used for probabilities.
- Unavailable, undefined, and zero-valued results are distinguished. For example, precision with no predicted positives receives an explanation rather than an unexplained success-looking zero.
- Unsupported graphs receive a specific reason and do not prevent unrelated valid results from being inspected.
- SHAP continues to identify the exact model output being explained. It is not presented as causation or as contributions in percentile-score points.
- Step 5 retains real stage and model logs. Any progress percentage represents observable completed work or explicitly labeled stage progress, not a fabricated estimate of elapsed training.
- Verification covers all seven requested metrics and the confusion matrix for the applicable saved model and operating point. The threshold explorer's results remain traceable to the same saved evaluation predictions.

## User flow

1. **Upload & Select:** upload a dataset and confirm its target, positive outcome, and column settings. Existing parsing and selection behavior remain in place.
2. **Clean Data:** retain the current cleaning workflow and reports.
3. **Explore Data:** retain the existing dataset-derived exploration views.
4. **Build Features:** retain training-only feature preparation and the reserved evaluation split.
5. **Train Models:** retain the training choices and show genuine model status, completed work, and logs. Successful completion makes the current run's evaluation available.
6. **Model Performance:** open Overview, inspect another trained model when desired, explore the ten tabs, adjust an evaluation threshold, and inspect verification evidence.
7. **Score & Rank Leads:** retain calibrated probabilities, training-reference percentile scores from 0–100, priority bands, and deterministic contact order from the saved scoring model.
8. **Explain a Lead:** retain real contributions for the chosen record and the identified scoring estimator.
9. **Export & Summary:** preserve downloads and summaries, ensuring evaluation-dependent outputs correspond to the active completed run.

Changing model-affecting selections returns the affected downstream steps to an incomplete state. Training is initiated through Step 5 rather than starting automatically while a target dropdown is being edited.

## UI/UX feel

- Preserve the existing light theme, typography, cards, colors, navigation, and nine-step structure.
- Add the requested tabs and model context within Step 6 rather than introducing a replacement dashboard or extra workflow step.
- Clearly distinguish the scoring winner, inspected model, saved threshold, and exploratory threshold.
- Use readable loading, recalculating, unavailable, invalidated, and failed states instead of blank plots, endless loading, or stale numbers.
- Keep chart labels readable across different feature names and dataset sizes; retain responsive layouts, keyboard-accessible controls, metric guidance, and reduced-motion behavior.

## Implementation phases

### Phase 1 — MVP: complete the requested targeted update

Preserve existing compliant functionality and implement only the identified gaps: ten-tab evaluation, all requested overview metrics, consistent model inspection, missing lift visualization, correct threshold interactions, adaptive chart states, stronger result invalidation and refresh, and expanded prediction-based verification. All requirements in this update belong to Phase 1; none is deferred to a later phase.

### Phase 2 — Optional future addition, not included now

Side-by-side comparison of separately completed historical runs. This would be an explicitly labeled history feature, not reuse of outdated results in the current workspace.

### Phase 3 — Optional future addition, not included now

Monitoring predictions against subsequently observed real-world outcomes and identifying performance drift over time.

Only Phase 1 is proposed for the current build. Phases 2 and 3 require separate approval and do not change the present workflow.

## Assumptions

1. This is a focused update to the existing application, not a rebuild. Working calculations and unrelated functionality remain unchanged unless a specific inconsistency requires correction.
2. The evaluation selector changes inspection only. The cross-validation-selected winner continues to power scoring and lead explanations.
3. Target and other configuration changes become authoritative when confirmed; unsaved edits block misleading downstream inspection. Confirmed changes require another Step 5 training run, not automatic background retraining on every dropdown change.
4. PR-AUC continues to mean average precision throughout selection, presentation, and verification.
5. The threshold explorer changes binary classification decisions, not probabilities or ranking. It does not overwrite the saved operational threshold.
6. Before/after calibration comparisons and uncertainty intervals are model-specific. Missing calculations are not borrowed from the winner or another run.
7. Additional model-specific learning and importance calculations may require time. Loading states are preferable to immediate but substituted results.
8. Mathematical references such as a no-skill diagonal, a lift baseline of one, and valid probability-axis bounds are allowed when explicitly identified. They are not purported model predictions. The PR prevalence reference must always use the actual evaluation labels.
9. Existing percentile scoring, priority boundaries, reproducibility behavior, and explanation limitations remain unchanged. Constants defining methodology are not hardcoded evaluation outcomes.
10. Different datasets may legitimately yield some identical metrics or baseline curves. Results must reflect the calculations, not be artificially altered to appear different.
11. The current retained binary-classification safeguards remain in force. Unsuitable data or unsupported explanations produce honest limitations rather than invented success.