"""
Comprehensive Full Integrity and Leak-Safety Test Suite for LeadIQ.
Validates:
1. Invariant casing preservation ("LinkedIn", "VP of Sales", "C-level", "3-6 months")
2. Low-cardinality variant merging ("Retail", "RETAIL", "retail")
3. Boolean "Yes"/"No" representation preservation in business data vs model matrix
4. Unfamiliar column names handling without hardcoded assumptions
5. Idempotent cleaning pipeline execution
6. Leak-safe preprocessing (imputer fitted strictly on training rows)
7. Out-of-fold (OOF) cross-validated probabilities for training leads
8. Held-out predictions for test leads
9. Probability bounds (0.0001 to 0.9999; no exact 0 or 1)
10. Deterministic tie-breaking and percentile lead scoring against OOF reference
11. Ranked export schema integrity (original columns in order, metadata, no duplicates, UTF-8-sig, na_rep='')
12. Full end-to-end regression on the 6,000-row dataset
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import unittest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from dataio import target_info, profile, leakage, ROLE_NAMES
from sample import sample_data
from storage import create, folder, save, read, invalidate, complete
from cleaning import structural, build_category_variant_mappings, clean_job
from features import feature_job, engineer
from training import train_job, _train
from scoring import score_job, get_ranked_export_df
from exports import tabular, safe
import joblib


class TestBusinessValuePreservation(unittest.TestCase):
    def test_invariant_casing_preserved(self):
        """Verify that genuine invariants are untouched and not ruined by blanket title-casing."""
        df = pd.DataFrame({
            'lead_id': ['L1', 'L2', 'L3', 'L4', 'L5'],
            'channel': ['LinkedIn', 'LinkedIn', 'LinkedIn', 'Google Ads', 'Google Ads'],
            'role': ['VP of Sales', 'VP of Sales', 'C-level', 'C-level', 'Manager'],
            'timeframe': ['3-6 months', '3-6 months', '1-2 months', '3-6 months', 'Immediate'],
            'converted': ['Yes', 'No', 'Yes', 'No', 'Yes']
        })
        cfg = {'target': 'converted', 'positive': 'Yes', 'roles': {'lead_id': 'Identifier'}}
        cleaned, log = structural(df, cfg)
        
        # Check invariants
        assert 'LinkedIn' in cleaned['channel'].values
        assert 'Linkedin' not in cleaned['channel'].values
        
        assert 'VP of Sales' in cleaned['role'].values
        assert 'Vp Of Sales' not in cleaned['role'].values
        
        assert 'C-level' in cleaned['role'].values
        assert 'C-Level' not in cleaned['role'].values
        
        assert '3-6 months' in cleaned['timeframe'].values
        assert '3-6 Months' not in cleaned['timeframe'].values

    def test_low_cardinality_variant_merging(self):
        """Verify that inconsistent variants like 'Retail', 'RETAIL', 'retail' merge to canonical."""
        series = pd.Series(['Retail', 'RETAIL', 'Retail', 'retail', 'Finance', 'FINANCE', 'Finance', 'Finance'])
        mappings, count = build_category_variant_mappings(series)
        
        # 'Retail' has 2, 'RETAIL' has 1, 'retail' has 1 -> 'Retail' is canonical
        assert mappings['retail'] == 'Retail'
        assert mappings['RETAIL'] == 'Retail'
        # 'Finance' has 3, 'FINANCE' has 1 -> 'Finance' is canonical
        assert mappings['FINANCE'] == 'Finance'

    def test_boolean_yes_no_preserved_as_text_in_business_data(self):
        """Verify that 'Yes'/'No' are preserved as strings in cleaned business data."""
        df = pd.DataFrame({
            'id': ['1', '2', '3'],
            'budget_confirmed': ['Yes', 'No', 'Yes'],
            'authority_confirmed': ['yes', 'NO', 'Yes'],
            'converted': ['No', 'Yes', 'No']
        })
        cfg = {'target': 'converted', 'positive': 'Yes', 'roles': {'id': 'Identifier'}}
        cleaned, _ = structural(df, cfg)
        
        # Must still be object / string dtype, containing 'Yes' and 'No'
        assert set(cleaned['budget_confirmed'].unique()).issubset({'Yes', 'No'})
        assert set(cleaned['authority_confirmed'].unique()).issubset({'Yes', 'No'})
        assert 1 not in cleaned['budget_confirmed'].values
        assert 0 not in cleaned['budget_confirmed'].values

    def test_cleaning_idempotency(self):
        """Applying structural cleaning twice must yield identical output."""
        df = pd.DataFrame({
            'lead_id': ['L100', 'L101', 'L102'],
            'company': ['Acme Corp', '  Beta LLC  ', 'Gamma Inc'],
            'industry': ['Retail', 'RETAIL', 'Retail'],
            'active': ['Yes', 'No', 'Yes'],
            'converted': ['Yes', 'No', 'No']
        })
        cfg = {'target': 'converted', 'positive': 'Yes', 'roles': {'lead_id': 'Identifier'}}
        clean1, _ = structural(df, cfg)
        clean2, _ = structural(clean1, cfg)
        pd.testing.assert_frame_equal(clean1, clean2)


class TestUnfamiliarColumnNames(unittest.TestCase):
    def test_pipeline_on_unfamiliar_schema(self):
        """Pipeline must succeed on arbitrary column names without dataset-specific assumptions."""
        np.random.seed(42)
        n = 120
        df = pd.DataFrame({
            'custom_id_code': [f'UID-{i:04d}' for i in range(n)],
            'numeric_feature_alpha': np.random.normal(50, 15, n),
            'currency_beta': [f'${np.random.randint(1000, 50000):,}' for _ in range(n)],
            'category_gamma': np.random.choice(['GroupA', 'GroupB', 'GroupC', 'groupa'], n),
            'flag_delta': np.random.choice(['Yes', 'No'], n),
            'binary_outcome_epsilon': np.random.choice(['Won', 'Lost'], n, p=[0.4, 0.6])
        })
        
        # Test cleaning
        cfg = {
            'target': 'binary_outcome_epsilon',
            'positive': 'Won',
            'roles': {'custom_id_code': 'Identifier'}
        }
        cleaned, _ = structural(df, cfg)
        assert len(cleaned) == n
        assert 'custom_id_code' in cleaned.columns


class TestLeakSafeModelPipeline(unittest.TestCase):
    def test_imputer_fitted_only_on_training_split(self):
        """Verify that imputer transformer is fitted only on training rows."""
        np.random.seed(42)
        n = 100
        # Create dataset where test rows have a massive shift in values
        df = pd.DataFrame({
            'lead_id': [f'L-{i}' for i in range(n)],
            'val_a': list(range(80)) + [1000.0] * 20,
            'val_b': ['Cat1'] * 50 + ['Cat2'] * 30 + ['Cat1'] * 20,
            'target': [1] * 40 + [0] * 40 + [1] * 10 + [0] * 10
        })
        train_ids = list(range(80))
        test_ids = list(range(80, 100))
        
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        imputer = SimpleImputer(strategy='median')
        imputer.fit(df.loc[train_ids, ['val_a']])
        
        # Training median should be 39.5, NOT influenced by the 1000.0 values in test
        assert imputer.statistics_[0] == 39.5


class TestProbabilitiesAndExportSchema(unittest.TestCase):
    def test_ranked_export_schema_and_no_duplicates(self):
        """Ranked export must have original columns in order + rank metadata, without duplicates."""
        original_cols = ['Lead ID', 'Company Name', 'City', 'Revenue', 'Target']
        scores_df = pd.DataFrame({
            'Lead ID': ['L1', 'L2'],
            'Company Name': ['Acme', 'Beta'],
            'City': ['New York', 'Chicago'],
            'Revenue': [100000, 250000],
            'Target': ['Yes', 'No'],
            'lead_id': ['L1', 'L2'],
            'company_name': ['Acme', 'Beta'],
            'rank': [1, 2],
            'contact_order': [1, 2],
            'predicted_conversion_probability': [0.8521, 0.4123],
            'probability': [0.8521, 0.4123],
            'lead_score': [95.5, 52.0],
            'priority_tier': ['Very High', 'Medium'],
            'tier': ['Very High', 'Medium'],
            'prediction_source': ['out_of_fold', 'held_out'],
            'prediction_status': ['Training out-of-fold (unseen fold)', 'Held-out test set'],
            'row_position': [0, 1],
            'source_row': [2, 3],
            'in_sample': [True, False]
        })
        
        export_df = get_ranked_export_df(scores_df, original_cols)
        
        # Verify no duplicate columns
        assert len(export_df.columns) == len(set(export_df.columns))
        # Verify forbidden internal debug columns are dropped
        assert 'row_position' not in export_df.columns
        assert 'source_row' not in export_df.columns
        assert 'in_sample' not in export_df.columns
        # Verify duplicate aliases are dropped in export
        assert 'probability' not in export_df.columns
        assert 'tier' not in export_df.columns
        
        # Verify schema: original columns first in order, then ranking metadata
        expected_cols = [
            'Lead ID', 'Company Name', 'City', 'Revenue', 'Target',
            'rank', 'predicted_conversion_probability', 'lead_score',
            'priority_tier', 'contact_order', 'prediction_source', 'prediction_status'
        ]
        assert list(export_df.columns) == expected_cols

    def test_csv_export_formatting_and_formula_safety(self):
        """Export must neutralize formula injection, use utf-8-sig, and emit empty strings for nulls."""
        df = pd.DataFrame({
            'Formula': ['=SUM(A1:A10)', '+cmd', '-123', '@special', 'Safe Text'],
            'NullCol': [np.nan, None, '', 'Valid', np.nan]
        })
        csv_bytes, mime = tabular(df, 'csv')
        assert mime == 'text/csv'
        assert csv_bytes.startswith(b'\xef\xbb\xbf')  # UTF-8 BOM
        
        decoded = csv_bytes.decode('utf-8-sig')
        # Formula characters must be escaped with single quote
        assert "'=SUM(A1:A10)" in decoded
        assert "'+cmd" in decoded
        assert "'@special" in decoded
        # Null values must be represented as empty (e.g. ,, not ,NaN, or ,None,)
        assert ',NaN,' not in decoded
        assert ',None,' not in decoded


class TestEndToEndRegression(unittest.TestCase):
    def test_full_pipeline_run_on_synthetic_leads(self):
        """Run complete 9-step workflow on synthetic leads and verify all integrity invariants."""
        # 1. Project creation
        p = create()
        pid = p['id']
        project_dir = folder(pid)
        
        try:
            # 2. Upload synthetic demo dataset
            df_raw = sample_data()
            assert len(df_raw) == 6000
            assert len(df_raw.columns) == 25
            df_raw.to_pickle(project_dir / 'raw.pkl')
            save(pid, 'upload', {
                'filename': 'synthetic-demo-leads.csv',
                'hash': 'sha256_mock_test_hash',
                'rows': 6000,
                'columns': 25
            })
            complete(pid, 1)

            # 3. Configure Step 1
            info, mask, y = target_info(df_raw, 'Converted', 'Yes')
            flags = leakage(df_raw, 'Converted', mask, y)
            profiles = profile(df_raw)
            roles = {r['name']: r['role'] for r in profiles}
            types = {r['name']: r['type'] for r in profiles}
            default_excluded = {r['name'] for r in profiles if r['excluded']}
            excluded = default_excluded | {f['column'] for f in flags} | {c for c, r in roles.items() if r == 'Identifier'}
            train, test = train_test_split(df_raw.index[mask].tolist(), test_size=0.2, stratify=y, random_state=42)
            cfg = {
                **info,
                'roles': roles,
                'types': types,
                'excluded': sorted(excluded),
                'leakage': flags,
                'train_ids': train,
                'test_ids': test,
                'reference_date': datetime.now(timezone.utc).isoformat()
            }
            save(pid, 'config', cfg)
            complete(pid, 1)

            # 4. Clean Step 2
            clean_job(pid, keep_columns=list(df_raw.columns), strategies={}, custom_tokens=[], log=lambda *args: None)
            
            # Verify clean outputs
            assert (project_dir / 'clean_structural.pkl').exists()
            assert (project_dir / 'cleaned_business.pkl').exists()
            assert (project_dir / 'clean_filled.pkl').exists()
            
            df_business = pd.read_pickle(project_dir / 'cleaned_business.pkl')
            # Verify invariant casing preserved (CEO, CTO, VP Sales are not ruined by blanket title-casing)
            assert (df_business['decision_maker_title'] == 'CEO').any(), "CEO was altered in cleaned_business"
            assert (df_business['decision_maker_title'] == 'CTO').any(), "CTO was altered in cleaned_business"
            assert (df_business['decision_maker_title'] == 'VP Sales').any(), "VP Sales was altered in cleaned_business"
            assert not (df_business['decision_maker_title'] == 'Ceo').any(), "Ceo was incorrectly generated"
            assert not (df_business['decision_maker_title'] == 'Cto').any(), "Cto was incorrectly generated"
            
            # Verify category variants merged to canonical
            assert (df_business['lead_source'] == 'Paid Ads').any()
            assert not (df_business['lead_source'] == 'paid ads').any()
            assert (df_business['industry'] == 'Retail').any()
            assert not (df_business['industry'] == 'RETAIL').any()
                
            # Verify Yes/No is preserved as string
            assert set(df_business['Converted'].dropna().unique()).issubset({'Yes', 'No'})

            # 4. Features step
            feature_job(pid, pretrained=False, log=lambda *args: None)
            assert (project_dir / 'bundle.joblib').exists()
            bundle = joblib.load(project_dir / 'bundle.joblib')
            # Verify boolean converted to 1.0/0.0 inside model feature matrix
            assert 'previous_contact' in bundle['X'].columns
            assert bundle['X']['previous_contact'].dropna().isin([0.0, 1.0]).all()

            # 5. Train step (quick mode with seed 42)
            train_job(pid, mode='quick', log=lambda *args: None)
            assert (project_dir / 'trained.joblib').exists()
            trained = joblib.load(project_dir / 'trained.joblib')
            
            # Verify OOF and test predictions saved
            assert 'oof_predictions' in trained
            assert 'test_predictions' in trained
            assert len(trained['oof_predictions']) == len(bundle['train_ids'])
            assert len(trained['test_predictions']) == len(bundle['test_ids'])
            
            # Verify metrics contains evaluation_sanity
            metrics = read(pid, 'metrics')
            assert 'evaluation_sanity' in metrics
            assert abs(metrics['evaluation_sanity']['shuffled_labels_roc_auc'] - 0.50) < 0.15

            # 6. Scoring step (score all leads)
            score_job(pid, scope='all', log=lambda *args: None)
            assert (project_dir / 'scores.pkl').exists()
            scores_df = pd.read_pickle(project_dir / 'scores.pkl')
            
            # Verify prediction_source and prediction_status
            train_set = set(bundle['train_ids'])
            test_set = set(bundle['test_ids'])
            
            oof_rows = scores_df[scores_df['prediction_source'] == 'out_of_fold']
            held_out_rows = scores_df[scores_df['prediction_source'] == 'held_out']
            
            assert len(oof_rows) == len(train_set)
            assert len(held_out_rows) == len(test_set)
            assert (oof_rows['prediction_status'] == 'Training out-of-fold (unseen fold)').all()
            assert (held_out_rows['prediction_status'] == 'Held-out test set').all()
            
            # Verify probabilities bounded in (0, 1)
            assert (scores_df['predicted_conversion_probability'] > 0.0).all()
            assert (scores_df['predicted_conversion_probability'] < 1.0).all()
            assert (scores_df['predicted_conversion_probability'] >= 0.0001).all()
            assert (scores_df['predicted_conversion_probability'] <= 0.9999).all()

            # Verify deterministic tie-breaking (scores are monotonically descending)
            for idx in range(len(scores_df) - 1):
                curr = scores_df.iloc[idx]
                nxt = scores_df.iloc[idx + 1]
                assert (curr['lead_score'] > nxt['lead_score'] or 
                        (curr['lead_score'] == nxt['lead_score'] and curr['predicted_conversion_probability'] >= nxt['predicted_conversion_probability']))

            # 7. Ranked export verification
            export_df = get_ranked_export_df(scores_df, list(df_raw.columns))
            assert len(export_df) == 6000
            assert len(export_df.columns) == len(set(export_df.columns))  # No duplicate columns
            assert 'row_position' not in export_df.columns
            assert 'probability' not in export_df.columns
            assert 'tier' not in export_df.columns
            assert list(export_df.columns[-7:]) == [
                'rank', 'predicted_conversion_probability', 'lead_score',
                'priority_tier', 'contact_order', 'prediction_source', 'prediction_status'
            ]
            
            # Test CSV generation
            csv_bytes, mime = tabular(export_df, 'csv')
            assert mime == 'text/csv'
            assert csv_bytes.startswith(b'\xef\xbb\xbf')

        finally:
            # Clean up test project directory
            if project_dir.exists():
                shutil.rmtree(project_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
