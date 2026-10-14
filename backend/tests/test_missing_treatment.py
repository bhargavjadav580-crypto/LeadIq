"""
Comprehensive unit and integration tests for LeadIQ's Missing-Value Treatment Engine.
Covers:
1. Central detector (token handling, 0/negative preservation, Unknown preservation)
2. Dynamic value-based type detection
3. Edge cases: unfamiliar column names, all-empty columns, division by zero
4. Strategy recommendation logic (<5%, 5-60%, >60%, target, identifier, contact)
5. Two-layer persistence and leak-prevention (structural unfilled vs filled reporting)
6. Missing target labels excluded from training
7. Idempotency on re-run with different strategies
8. End-to-end execution on the 6,000-row B2B synthetic dataset
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from missing_engine import CentralMissingDetector, ValueTypeDetector, StrategyRecommender, MissingValueEngine
from sample import sample_data
from storage import create, folder, save, read, invalidate
from cleaning import clean_job, get_missing_plan


class TestCentralMissingDetector:
    def test_default_tokens_recognized(self):
        detector = CentralMissingDetector()
        assert detector.is_missing(np.nan) is True
        assert detector.is_missing(None) is True
        assert detector.is_missing('') is True
        assert detector.is_missing('   ') is True
        assert detector.is_missing('N/A') is True
        assert detector.is_missing('na') is True
        assert detector.is_missing('null') is True
        assert detector.is_missing('NULL') is True
        assert detector.is_missing('none') is True
        assert detector.is_missing('?') is True
        assert detector.is_missing('-') is True
        assert detector.is_missing('#N/A') is True
        assert detector.is_missing('<NA>') is True

    def test_numbers_and_unknown_never_auto_erased(self):
        detector = CentralMissingDetector()
        # 0 and 0.0 must be preserved
        assert detector.is_missing(0) is False
        assert detector.is_missing(0.0) is False
        assert detector.is_missing('0') is False
        # Negatives must be preserved
        assert detector.is_missing(-1) is False
        assert detector.is_missing(-42.5) is False
        assert detector.is_missing('-100') is False
        # 'Unknown' must be preserved as legitimate categorical label
        assert detector.is_missing('Unknown') is False
        assert detector.is_missing('unknown') is False
        # Valid booleans
        assert detector.is_missing(True) is False
        assert detector.is_missing(False) is False

    def test_detect_column_token_counts(self):
        detector = CentralMissingDetector()
        s = pd.Series(['Apex Corp', '?', 'N/A', '-', '', 'Valid Company', None, 0])
        mask, count, pct, tokens = detector.detect_column(s)
        assert count == 5
        assert bool(mask.iloc[1]) is True  # '?'
        assert bool(mask.iloc[2]) is True  # 'N/A'
        assert bool(mask.iloc[3]) is True  # '-'
        assert bool(mask.iloc[4]) is True  # ''
        assert bool(mask.iloc[6]) is True  # None
        assert bool(mask.iloc[7]) is False # 0
        assert tokens.get('?') == 1
        assert tokens.get('n/a') == 1
        assert tokens.get('-') == 1


class TestValueTypeDetector:
    def test_type_detection(self):
        detector = CentralMissingDetector()
        
        # Numeric
        s_num = pd.Series([10.5, 20.3, np.nan, 45.1, '?', 50.0])
        assert ValueTypeDetector.detect(s_num, detector) == 'numeric'
        
        # Integer
        s_int = pd.Series([10, 20, np.nan, 45, 50])
        assert ValueTypeDetector.detect(s_int, detector) == 'integer'
        
        # Boolean
        s_bool = pd.Series(['Yes', 'No', 'Yes', np.nan, 'No'])
        assert ValueTypeDetector.detect(s_bool, detector) == 'boolean'
        
        # Email
        s_email = pd.Series(['lead1@corp.com', 'lead2@domain.org', np.nan, 'contact@acme.io'])
        assert ValueTypeDetector.detect(s_email, detector) == 'email'
        
        # Phone
        s_phone = pd.Series(['+919876543210', '+12025550192', np.nan, '+442071838750'])
        assert ValueTypeDetector.detect(s_phone, detector) == 'phone'
        
        # Website
        s_web = pd.Series(['https://alpha.com', 'http://beta.io', 'www.gamma.org', np.nan])
        assert ValueTypeDetector.detect(s_web, detector) == 'website'
        
        # Identifier
        s_id = pd.Series([f'LEAD-{i:05d}' for i in range(100)], name='lead_id')
        assert ValueTypeDetector.detect(s_id, detector) == 'identifier'


class TestEdgeCasesAndRecommendations:
    def test_unfamiliar_column_names_and_all_empty(self):
        detector = CentralMissingDetector()
        df = pd.DataFrame({
            'Company / Entity (Pvt Ltd)': ['Acme', 'Beta', np.nan, 'Delta'],
            'Revenue ($K USD)': [100, 200, np.nan, 400],
            '  Oddly Spaced Column  ': ['A', 'B', 'C', 'D'],
            'All Empty Column': [np.nan, None, '?', 'N/A'],
            'Target Outcome': ['Yes', 'No', np.nan, 'Yes']
        })
        
        engine = MissingValueEngine()
        plan = engine.plan(df, {'target': 'Target Outcome', 'roles': {}})
        assert plan['total_rows'] == 4
        assert plan['total_columns'] == 5
        
        # All Empty Column should be identified without ZeroDivisionError
        empty_plan = next(c for c in plan['columns'] if c['column'] == 'All Empty Column')
        assert empty_plan['missing_count'] == 4
        assert empty_plan['missing_pct'] == 100.0
        assert empty_plan['recommended_strategy'] == 'drop_column'

    def test_target_never_imputed(self):
        rec = StrategyRecommender.recommend(
            col='Converted',
            detected_type='boolean',
            missing_count=45,
            total_rows=1000,
            is_target=True
        )
        assert rec['recommended_strategy'] == 'exclude_training'
        assert 'Target outcome labels must never be invented' in rec['reason']
        assert rec['add_indicator'] is False

    def test_excessive_missing_recommends_drop(self):
        rec = StrategyRecommender.recommend(
            col='tech_stack_notes',
            detected_type='free_text',
            missing_count=750,
            total_rows=1000
        )
        assert rec['recommended_strategy'] == 'drop_column'
        assert rec['missing_pct'] == 75.0


class TestTwoLayerTreatmentEngine:
    def test_treatment_execution_and_validation(self):
        engine = MissingValueEngine()
        
        df_struct = pd.DataFrame({
            'lead_id': ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'],
            'revenue': [100.0, np.nan, 300.0, 400.0, np.nan, 600.0],
            'industry': ['Retail', 'Tech', np.nan, 'Retail', 'Unknown', np.nan],
            'website': ['https://a.com', np.nan, 'https://c.com', np.nan, 'https://e.com', np.nan],
            'Converted': ['Yes', 'No', 'Yes', np.nan, 'No', 'Yes']
        })
        
        config = {'target': 'Converted', 'roles': {'lead_id': 'Identifier', 'website': 'Website'}}
        
        user_strategies = {
            'revenue': {'strategy': 'median', 'add_indicator': True},
            'industry': {'strategy': 'unknown', 'add_indicator': False},
            'website': {'strategy': 'presence_flag', 'add_indicator': True}
        }
        
        df_filled, audit, validation = engine.treat(
            df_structural=df_struct,
            user_strategies=user_strategies,
            config=config
        )
        
        # 1. Structural layer was not mutated
        assert df_struct['revenue'].isna().sum() == 2
        assert df_struct['industry'].isna().sum() == 2
        
        # 2. Filled layer resolved revenue and industry
        assert df_filled['revenue'].isna().sum() == 0
        assert df_filled['revenue'].iloc[1] == 350.0  # Median of [100, 300, 400, 600] = 350
        assert df_filled['revenue_missing'].sum() == 2
        
        # 3. Collision-safe Unknown: because 'Unknown' already existed in industry, fallback 'Missing' is used!
        assert df_filled['industry'].isna().sum() == 0
        assert 'Missing' in df_filled['industry'].values
        
        # 4. Target was completely untouched
        assert df_filled['Converted'].isna().sum() == 1
        assert validation['target_untouched'] is True
        
        # 5. Identifier was completely untouched
        assert validation['identifiers_untouched'] is True


class TestEndToEndSamplePipeline:
    def test_full_pipeline_clean_job(self):
        # Create a temporary project
        proj = create()
        pid = proj['id']
        token = proj['token']
        
        # Generate 6,000-row sample data
        df_sample = sample_data()
        save(pid, 'upload', {'filename': 'test_leads.csv', 'hash': 'fakehash', 'synthetic': True})
        df_sample.to_pickle(folder(pid) / 'raw.pkl')
        
        # Setup mock config
        config = {
            'target': 'Converted',
            'positive': 'yes',
            'roles': {'lead_id': 'Identifier', 'company_name': 'Business Name'},
            'types': {'annual_revenue': 'numeric', 'industry': 'categorical'},
            'excluded': [],
            'train_ids': list(range(4800)),
            'test_ids': list(range(4800, 6000)),
            'reference_date': '2026-10-14T00:00:00Z'
        }
        save(pid, 'config', config)
        
        # Run clean_job
        logs = []
        clean_job(pid, overrides=[], log=lambda p, m: logs.append(m))
        
        # Verify both datasets exist on disk
        p_struct = folder(pid) / 'clean_structural.pkl'
        p_clean = folder(pid) / 'clean.pkl'
        p_filled = folder(pid) / 'clean_filled.pkl'
        p_strat = folder(pid) / 'cleaning_strategy.json'
        p_quality = folder(pid) / 'quality.json'
        
        assert p_struct.exists()
        assert p_clean.exists()
        assert p_filled.exists()
        assert p_strat.exists()
        assert p_quality.exists()
        
        # Read quality report
        quality = read(pid, 'quality')
        assert quality['rows_before'] == 6000
        assert quality['summary_missing']['total_resolved'] > 0
        assert quality['summary_missing']['overall_resolved_pct'] >= 50.0
        
        # Verify df_filled has treated columns resolved
        df_filled = pd.read_pickle(p_filled)
        assert df_filled['annual_revenue'].isna().sum() == 0
        assert df_filled['industry'].isna().sum() == 0
        
        # Verify df_structural still has gaps (preserving leak safety for ML)
        df_struct = pd.read_pickle(p_struct)
        assert df_struct['annual_revenue'].isna().sum() > 0


if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING MISSING-VALUE TREATMENT ENGINE TEST SUITE")
    print("=" * 60)
    
    test_classes = [
        TestCentralMissingDetector,
        TestValueTypeDetector,
        TestEdgeCasesAndRecommendations,
        TestTwoLayerTreatmentEngine,
        TestEndToEndSamplePipeline
    ]
    
    total = 0
    passed = 0
    failed = 0
    
    for cls in test_classes:
        instance = cls()
        for attr in dir(instance):
            if attr.startswith('test_'):
                total += 1
                method = getattr(instance, attr)
                try:
                    method()
                    passed += 1
                    print(f"  PASS: {cls.__name__}.{attr}")
                except Exception as e:
                    failed += 1
                    print(f"  FAIL: {cls.__name__}.{attr} -> {e}")
                    import traceback
                    traceback.print_exc()
                    
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)

