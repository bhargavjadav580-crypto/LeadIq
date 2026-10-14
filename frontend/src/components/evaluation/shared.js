export const metricsList=[['accuracy','Accuracy'],['precision','Precision'],['recall','Recall'],['f1','F1 score'],['roc_auc','ROC-AUC'],['pr_auc','PR-AUC'],['precision_top10','Precision @ top 10%'],['brier_score','Brier Score']];
export const formatMetric=v=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(3):'Unavailable';
export const tabs=['Overview','Model Comparison','ROC Curve','Precision-Recall Curve','Confusion Matrix','Learning Curve','Lift & Gain','Calibration','Feature Importance','Verification'];
export const tabId=t=>t.toLowerCase().replace(/[^a-z]+/g,'-');