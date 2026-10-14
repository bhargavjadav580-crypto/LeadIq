"""
LeadIQ Missing-Value Treatment Engine
Detects, plans, treats, validates, and persists missing value transformations.
Preserves strict leak-safety between Layer 1 (user-visible reporting/export)
and Layer 2 (scikit-learn training pipeline).
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set

# Shared default missing tokens
# Note: '0', negative numbers, and 'Unknown'/'unknown' are NEVER treated as missing automatically!
DEFAULT_MISSING_TOKENS: Set[str] = {
    '', ' ', 'nan', 'na', 'n/a', 'null', 'none', '-', '?', '#n/a', '<na>', 'nil', 'undefined'
}


class CentralMissingDetector:
    """
    Central shared detector for missing values across all LeadIQ workflows.
    Detects NaN, None, empty string, whitespace-only strings, and configurable token list.
    Safely preserves 0, negative numbers, and 'Unknown'.
    """

    def __init__(self, custom_tokens: Optional[List[str]] = None):
        if custom_tokens is not None:
            self.tokens = {str(t).strip().lower() for t in custom_tokens if str(t).strip().lower() not in {'0', 'unknown'}}
        else:
            self.tokens = set(DEFAULT_MISSING_TOKENS)

    def is_missing(self, val: Any) -> bool:
        if val is None or pd.isna(val) or val is pd.NA:
            return True
        if isinstance(val, (int, float, np.integer, np.floating)):
            # 0, 0.0, and negative numbers are genuine numerical quantities
            return False
        if isinstance(val, (bool, np.bool_)):
            return False
        s = str(val).strip()
        if s == '':
            return True
        if s.lower() in self.tokens:
            return True
        return False

    def detect_column(self, series: pd.Series) -> Tuple[pd.Series, int, float, Dict[str, int]]:
        """
        Inspect a Series and return:
        - boolean Series mask (True where missing)
        - count of missing values
        - percentage of missing values (0.0 to 100.0)
        - dictionary of exact tokens found and treated as missing
        """
        n = len(series)
        if n == 0:
            return pd.Series([], dtype=bool), 0, 0.0, {}

        # 1. Base null check
        base_null = series.isna() | series.isnull()
        
        # 2. Token check for non-numeric/string values
        token_mask = pd.Series(False, index=series.index)
        tokens_found: Dict[str, int] = {}

        # Scan string/object entries
        for idx, val in series.items():
            if base_null.loc[idx]:
                continue
            if isinstance(val, (int, float, np.integer, np.floating, bool, np.bool_)):
                continue
            s = str(val).strip()
            if s == '':
                token_mask.loc[idx] = True
                tokens_found['<empty>'] = tokens_found.get('<empty>', 0) + 1
            elif s.lower() in self.tokens:
                token_mask.loc[idx] = True
                tok = s.lower()
                tokens_found[tok] = tokens_found.get(tok, 0) + 1

        combined_mask = base_null | token_mask
        missing_count = int(combined_mask.sum())
        missing_pct = round((missing_count / n) * 100, 2) if n > 0 else 0.0

        return combined_mask, missing_count, missing_pct, tokens_found

    def standardize_series(self, series: pd.Series) -> pd.Series:
        """
        Returns a copy of the series where any detected missing token is normalized to np.nan.
        """
        mask, _, _, _ = self.detect_column(series)
        s = series.copy()
        s[mask] = np.nan
        return s


class ValueTypeDetector:
    """
    Detects column types dynamically from actual values rather than superficial column names.
    Supports user overrides.
    """

    @staticmethod
    def detect(series: pd.Series, detector: CentralMissingDetector, role: str = 'Automatic') -> str:
        # Check non-missing values
        mask, _, _, _ = detector.detect_column(series)
        valid = series[~mask]
        n_valid = len(valid)

        if n_valid == 0:
            return 'all_missing'

        sample = valid.astype(str).str.strip().head(1000)
        n_sample = len(sample)

        # 1. Check boolean
        lower_sample = set(sample.str.lower())
        bool_tokens = {'yes', 'no', 'y', 'n', 'true', 'false', '1', '0'}
        if lower_sample and lower_sample.issubset(bool_tokens) and len(lower_sample) <= 2:
            return 'boolean'

        col_name = str(series.name).lower() if series.name is not None else ''

        # 2. Check identifier patterns from role and name
        is_id_name = bool(re.search(r'(^id$|[_ ]id$|uuid|identifier|^code$|^lead_?id$|^customer_?id$)', col_name))
        if role == 'Identifier' or (is_id_name and valid.nunique() > 10):
            return 'identifier'

        # 3. Check contact & free text formats
        email_match = sample.str.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$').mean()
        if role == 'Email' or email_match > 0.7:
            return 'email'

        web_match = sample.str.match(r'^(?:https?://|www\.)', case=False).mean()
        if role == 'Website' or web_match > 0.7:
            return 'website'

        is_phone_name = 'phone' in col_name or 'mobile' in col_name or 'tel' in col_name
        phone_match = sample.str.match(r'^\+\d[\d\s().-]{7,}$').mean()
        if role == 'Phone' or phone_match > 0.7 or (is_phone_name and sample.str.match(r'^\+?[\d\s().-]{7,}$').mean() > 0.7):
            return 'phone'

        if role in ['Text', 'Review Text'] or (sample.str.len().mean() > 65 and sample.nunique() > 20):
            return 'free_text'

        # 4. Check numeric
        def try_num(val):
            try:
                s = str(val).strip().replace(',', '')
                m = re.fullmatch(r'[\$€£₹]?\s*([+-]?\d*\.?\d+)\s*([kmb%]?)(?:\s*(?:stars?|years?|reviews?))?', s, re.I)
                if m and m[1] != '':
                    return float(m[1])
                return np.nan
            except Exception:
                return np.nan

        num_parsed = sample.map(try_num)
        numeric_ratio = num_parsed.notna().mean()
        if role in ['Numeric', 'Rating', 'Review Count', 'Business Age'] or numeric_ratio > 0.85:
            valid_nums = num_parsed.dropna()
            if len(valid_nums) > 0 and (valid_nums % 1 == 0).mean() > 0.95:
                return 'integer'
            return 'numeric'

        # 5. Check high-uniqueness identifier
        if n_valid > 30 and (valid.nunique() / n_valid) > 0.98 and not sample.str.replace('.', '', regex=False).str.isdigit().all():
            return 'identifier'

        # 5. Check datetime
        date_pattern_match = sample.str.match(r'^\d{4}[-/]\d{1,2}[-/]').mean()
        if role == 'Datetime' or date_pattern_match > 0.7:
            return 'datetime'
        if 'date' in col_name or 'time' in col_name or 'timestamp' in col_name:
            try:
                parsed = pd.to_datetime(sample.head(50), errors='coerce', utc=True)
                if parsed.notna().mean() > 0.75:
                    return 'datetime'
            except Exception:
                pass

        # 6. Check mixed vs categorical
        if 0.15 < numeric_ratio < 0.85:
            return 'mixed'

        # Categorical cardinality
        n_unique = valid.nunique()
        if n_unique <= 30:
            return 'categorical_low'
        return 'categorical_high'


class StrategyRecommender:
    """
    Recommends optimal missing-value treatment strategies based on detected data type,
    cardinality, missingness severity, and distribution.
    """

    @staticmethod
    def recommend(
        col: str,
        detected_type: str,
        missing_count: int,
        total_rows: int,
        is_target: bool = False,
        is_identifier: bool = False,
        series: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0.0

        # Special Case 1: Target Variable
        if is_target:
            return {
                'column': col,
                'detected_type': detected_type,
                'missing_count': missing_count,
                'missing_pct': round(missing_pct, 2),
                'recommended_strategy': 'exclude_training' if missing_count > 0 else 'none',
                'reason': 'Target outcome labels must never be invented or imputed. Unlabeled leads will be excluded from model training, but preserved for operational scoring.' if missing_count > 0 else 'Target outcome variable has no missing values.',
                'alternatives': ['exclude_training', 'keep_missing'],
                'add_indicator': False,
                'status': 'Excluded from Training' if missing_count > 0 else 'Ready',
                'is_target': True,
                'is_identifier': False
            }

        # Special Case 2: Identifier
        if is_identifier or detected_type == 'identifier':
            return {
                'column': col,
                'detected_type': 'identifier',
                'missing_count': missing_count,
                'missing_pct': round(missing_pct, 2),
                'recommended_strategy': 'keep_missing',
                'reason': 'Unique identifier: synthetic IDs must never be fabricated. Excluded from machine learning features.',
                'alternatives': ['keep_missing', 'drop_column'],
                'add_indicator': False,
                'status': 'Preserved Unfilled' if missing_count > 0 else 'Ready',
                'is_target': False,
                'is_identifier': True
            }

        # Special Case 3: Zero missing values
        if missing_count == 0:
            return {
                'column': col,
                'detected_type': detected_type,
                'missing_count': 0,
                'missing_pct': 0.0,
                'recommended_strategy': 'none',
                'reason': 'No missing values detected. Complete data.',
                'alternatives': ['none', 'drop_column'],
                'add_indicator': False,
                'status': 'Complete',
                'is_target': False,
                'is_identifier': False
            }

        # Special Case 4: Excessive missingness (> 60%)
        if missing_pct > 60.0:
            return {
                'column': col,
                'detected_type': detected_type,
                'missing_count': missing_count,
                'missing_pct': round(missing_pct, 2),
                'recommended_strategy': 'drop_column',
                'reason': f'Excessive missingness ({missing_pct:.1f}% > 60%). Recommend dropping to avoid injecting excessive synthetic noise (overridable).',
                'alternatives': ['drop_column', 'median', 'mean', 'unknown', 'most_frequent', 'keep_missing'],
                'add_indicator': False,
                'status': 'Recommended Drop',
                'is_target': False,
                'is_identifier': False
            }

        # Case 5: Contact & Free Text
        if detected_type in ['email', 'website', 'phone', 'free_text']:
            return {
                'column': col,
                'detected_type': detected_type,
                'missing_count': missing_count,
                'missing_pct': round(missing_pct, 2),
                'recommended_strategy': 'presence_flag',
                'reason': 'Contact and free text details should not be fabricated. Keeps original missingness and creates an informative presence indicator flag.',
                'alternatives': ['presence_flag', 'keep_missing', 'drop_column'],
                'add_indicator': True,
                'status': 'Presence Flag',
                'is_target': False,
                'is_identifier': False
            }

        # Case 6: Numeric / Integer Features
        if detected_type in ['numeric', 'integer']:
            # Calculate skewness to choose between median and mean
            skew = 0.0
            if series is not None:
                try:
                    num_s = pd.to_numeric(series.dropna().astype(str).str.replace(',', '', regex=False), errors='coerce').dropna()
                    if len(num_s) > 2:
                        skew = float(num_s.skew())
                except Exception:
                    pass

            # Near-symmetric if abs(skew) < 0.8
            near_symmetric = abs(skew) < 0.8
            default_strat = 'mean' if near_symmetric else 'median'
            strat_reason = (
                f'Near-symmetric distribution (skewness: {skew:.2f}). Mean imputation preserves the expected value.'
                if near_symmetric else
                f'Skewed distribution (skewness: {skew:.2f}). Median imputation is robust against outliers.'
            )

            # Recommend indicator if missingness is between 5% and 60%
            add_ind = 5.0 <= missing_pct <= 60.0

            return {
                'column': col,
                'detected_type': detected_type,
                'missing_count': missing_count,
                'missing_pct': round(missing_pct, 2),
                'recommended_strategy': default_strat,
                'reason': strat_reason + (' Missing indicator recommended (5%–60% missing).' if add_ind else ''),
                'alternatives': ['median', 'mean', 'zero', 'constant', 'knn', 'iterative', 'drop_column', 'keep_missing'],
                'add_indicator': add_ind,
                'status': 'Treated',
                'is_target': False,
                'is_identifier': False
            }

        # Case 7: Categorical Features
        if detected_type in ['categorical_low', 'categorical_high']:
            if detected_type == 'categorical_high':
                return {
                    'column': col,
                    'detected_type': detected_type,
                    'missing_count': missing_count,
                    'missing_pct': round(missing_pct, 2),
                    'recommended_strategy': 'group_rare_unknown',
                    'reason': 'High cardinality categorical. Groups infrequent categories (<1% frequency) into "Other" and fills missing with collision-safe "Unknown".',
                    'alternatives': ['group_rare_unknown', 'unknown', 'most_frequent', 'drop_column', 'keep_missing'],
                    'add_indicator': 5.0 <= missing_pct <= 60.0,
                    'status': 'Treated',
                    'is_target': False,
                    'is_identifier': False
                }
            else:
                strat = 'most_frequent' if missing_pct < 2.0 else 'unknown'
                return {
                    'column': col,
                    'detected_type': detected_type,
                    'missing_count': missing_count,
                    'missing_pct': round(missing_pct, 2),
                    'recommended_strategy': strat,
                    'reason': f"Fills missing values with explicit 'Unknown' category to preserve missingness as an informative signal without label ambiguity.",
                    'alternatives': ['unknown', 'most_frequent', 'drop_column', 'keep_missing'],
                    'add_indicator': 5.0 <= missing_pct <= 60.0,
                    'status': 'Treated',
                    'is_target': False,
                    'is_identifier': False
                }

        # Case 8: Boolean Features
        if detected_type == 'boolean':
            return {
                'column': col,
                'detected_type': detected_type,
                'missing_count': missing_count,
                'missing_pct': round(missing_pct, 2),
                'recommended_strategy': 'most_frequent',
                'reason': 'Never assume missing boolean means False. Imputes the most frequent class (mode) while preserving binary semantics.',
                'alternatives': ['most_frequent', 'unknown', 'drop_column', 'keep_missing'],
                'add_indicator': 5.0 <= missing_pct <= 60.0,
                'status': 'Treated',
                'is_target': False,
                'is_identifier': False
            }

        # Case 9: Datetime
        if detected_type == 'datetime':
            return {
                'column': col,
                'detected_type': detected_type,
                'missing_count': missing_count,
                'missing_pct': round(missing_pct, 2),
                'recommended_strategy': 'leave_missing',
                'reason': 'Datetime timestamps are left missing to prevent forward chronological leakage. Derived feature engineering handles missing dates safely.',
                'alternatives': ['leave_missing', 'mode', 'drop_column'],
                'add_indicator': False,
                'status': 'Preserved Unfilled',
                'is_target': False,
                'is_identifier': False
            }

        # Case 10: Mixed / Ambiguous
        return {
            'column': col,
            'detected_type': 'mixed',
            'missing_count': missing_count,
            'missing_pct': round(missing_pct, 2),
            'recommended_strategy': 'unknown',
            'reason': 'Mixed text and numeric values detected. Treated as categorical with "Unknown" placeholder for missing entries.',
            'alternatives': ['unknown', 'drop_column', 'keep_missing'],
            'add_indicator': False,
            'status': 'Flagged Review',
            'is_target': False,
            'is_identifier': False
        }


class MissingValueEngine:
    """
    Main orchestrator for planning, applying, and validating missing-value treatments.
    """

    def __init__(self, custom_tokens: Optional[List[str]] = None):
        self.detector = CentralMissingDetector(custom_tokens)

    def plan(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a comprehensive Missing-Value Treatment Plan for every column in the dataset.
        """
        target_col = config.get('target')
        roles = config.get('roles', {})
        total_rows = len(df)
        total_cols = len(df.columns)

        columns_plan = []
        all_tokens_found: Dict[str, int] = {}
        total_missing = 0

        for col in df.columns:
            series = df[col]
            mask, missing_cnt, missing_pct, tokens = self.detector.detect_column(series)
            total_missing += missing_cnt
            for t, cnt in tokens.items():
                all_tokens_found[t] = all_tokens_found.get(t, 0) + cnt

            col_role = roles.get(col, 'Automatic')
            is_target = (col == target_col)
            is_id = (col_role == 'Identifier')

            detected_type = ValueTypeDetector.detect(series, self.detector, col_role)
            rec = StrategyRecommender.recommend(
                col=col,
                detected_type=detected_type,
                missing_count=missing_cnt,
                total_rows=total_rows,
                is_target=is_target,
                is_identifier=is_id,
                series=series
            )
            rec['tokens_found'] = tokens
            columns_plan.append(rec)

        overall_missing_pct = round((total_missing / (total_rows * total_cols) * 100), 2) if (total_rows * total_cols) > 0 else 0.0

        return {
            'total_rows': total_rows,
            'total_columns': total_cols,
            'total_missing': total_missing,
            'overall_missing_pct': overall_missing_pct,
            'tokens_treated_as_missing': sorted(list(all_tokens_found.keys())),
            'columns': columns_plan
        }

    def treat(
        self,
        df_structural: pd.DataFrame,
        user_strategies: Dict[str, Any],
        config: Dict[str, Any],
        raw_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """
        Applies configured missing-value treatments to produce df_filled (Layer 1).
        Preserves original df_structural completely untouched.
        Returns: (df_filled, audit_report, validation_report)
        """
        df_filled = df_structural.copy()
        target_col = config.get('target')
        roles = config.get('roles', {})

        audit_entries = []
        created_indicators = []
        dropped_cols = []
        treated_count = 0
        total_missing_before = 0
        total_missing_after = 0

        # Plan defaults
        plan = self.plan(df_structural, config)
        plan_lookup = {c['column']: c for c in plan['columns']}

        for col in df_structural.columns:
            orig_col = df_structural[col]
            col_plan = plan_lookup.get(col, {})
            mask, missing_cnt_struct, _, _ = self.detector.detect_column(orig_col)
            
            # Count raw missing if provided
            raw_missing_cnt = missing_cnt_struct
            if raw_df is not None and col in raw_df:
                raw_mask, raw_missing_cnt, _, _ = self.detector.detect_column(raw_df[col])
            total_missing_before += raw_missing_cnt

            is_target = (col == target_col)
            is_id = (roles.get(col) == 'Identifier' or col_plan.get('detected_type') == 'identifier')

            # User configuration overrides
            strat_cfg = user_strategies.get(col, {})
            if isinstance(strat_cfg, str):
                strategy = strat_cfg
                add_indicator = col_plan.get('add_indicator', False)
                const_val = None
            else:
                strategy = strat_cfg.get('strategy', col_plan.get('recommended_strategy', 'none'))
                add_indicator = strat_cfg.get('add_indicator', col_plan.get('add_indicator', False))
                const_val = strat_cfg.get('constant_value', None)

            # Never impute target or identifier
            if is_target:
                strategy = 'exclude_training'
                add_indicator = False
            elif is_id:
                strategy = 'keep_missing'
                add_indicator = False

            # Add indicator if configured and there were missing values
            indicator_col_name = None
            if add_indicator and missing_cnt_struct > 0 and not is_target and not is_id:
                indicator_col_name = f"{col}_missing"
                df_filled[indicator_col_name] = mask.astype(int)
                created_indicators.append(indicator_col_name)

            applied_value = None
            status = 'Preserved'

            if strategy == 'drop_column':
                dropped_cols.append(col)
                status = 'Dropped'
            elif missing_cnt_struct == 0 or strategy in ['none', 'leave_missing', 'keep_missing', 'exclude_training']:
                status = 'Complete' if missing_cnt_struct == 0 else 'Preserved Unfilled'
            elif strategy == 'presence_flag':
                # Retain missing, ensure presence flag exists
                flag_col = f"{col}_present"
                df_filled[flag_col] = (~mask).astype(int)
                if flag_col not in created_indicators:
                    created_indicators.append(flag_col)
                status = 'Presence Flag Created'
            elif strategy in ['median', 'mean', 'zero', 'constant', 'knn', 'iterative']:
                # Numeric treatment
                # Parse series to float
                s_numeric = pd.to_numeric(df_filled[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
                valid_vals = s_numeric[~mask]
                
                if strategy == 'median':
                    fill_val = float(valid_vals.median()) if len(valid_vals) > 0 else 0.0
                elif strategy == 'mean':
                    fill_val = float(valid_vals.mean()) if len(valid_vals) > 0 else 0.0
                elif strategy == 'zero':
                    fill_val = 0.0
                elif strategy == 'constant':
                    try:
                        fill_val = float(const_val) if const_val is not None else 0.0
                    except Exception:
                        fill_val = 0.0
                elif strategy in ['knn', 'iterative']:
                    # KNN / Iterative Imputation with strict safety guards
                    if len(df_filled) <= 10000 and len(df_filled.select_dtypes(include='number').columns) <= 30:
                        try:
                            if strategy == 'knn':
                                from sklearn.impute import KNNImputer
                                imp = KNNImputer(n_neighbors=min(5, max(1, len(valid_vals))))
                            else:
                                from sklearn.experimental import enable_iterative_imputer
                                from sklearn.impute import IterativeImputer
                                imp = IterativeImputer(max_iter=10, random_state=42)
                            num_cols = df_filled.select_dtypes(include='number').columns.tolist()
                            if col not in num_cols:
                                num_cols.append(col)
                                df_filled[col] = s_numeric
                            imputed_arr = imp.fit_transform(df_filled[num_cols])
                            col_idx = num_cols.index(col)
                            df_filled[col] = imputed_arr[:, col_idx]
                            fill_val = f'ML Imputed ({strategy.upper()})'
                        except Exception:
                            fill_val = float(valid_vals.median()) if len(valid_vals) > 0 else 0.0
                            df_filled.loc[mask, col] = fill_val
                    else:
                        # Fallback to median for safety on large dimensions
                        fill_val = float(valid_vals.median()) if len(valid_vals) > 0 else 0.0
                        df_filled.loc[mask, col] = fill_val
                
                if strategy not in ['knn', 'iterative'] or isinstance(fill_val, (int, float)):
                    if col_plan.get('detected_type') == 'integer' or (len(valid_vals) > 0 and (valid_vals % 1 == 0).all()):
                        fill_val = round(fill_val)
                    # Convert column to object or numeric so StringDtype assignment never errors
                    df_filled[col] = pd.to_numeric(df_filled[col], errors='coerce')
                    df_filled.loc[mask, col] = fill_val

                applied_value = fill_val
                status = 'Treated'
                treated_count += 1

            elif strategy in ['unknown', 'most_frequent', 'group_rare_unknown']:
                # Categorical treatment
                valid_cats = df_filled.loc[~mask, col].astype(str)
                existing_unique = set(valid_cats.unique())

                if strategy == 'group_rare_unknown':
                    # Group infrequent values (< 1%) into "Other"
                    freqs = valid_cats.value_counts(normalize=True)
                    rare = freqs[freqs < 0.01].index.tolist()
                    if rare:
                        df_filled[col] = df_filled[col].replace(rare, 'Other')

                if strategy == 'most_frequent':
                    mode_val = valid_cats.mode().iloc[0] if len(valid_cats) > 0 else 'Unknown'
                    df_filled[col] = df_filled[col].astype(object)
                    df_filled.loc[mask, col] = mode_val
                    applied_value = mode_val
                else:
                    # Collision prevention: if 'Unknown' already exists, use 'Missing' or 'Unknown_Value'
                    fill_label = 'Unknown'
                    if 'Unknown' in existing_unique:
                        fill_label = 'Missing'
                        if 'Missing' in existing_unique:
                            fill_label = 'Unknown_Value'
                    df_filled[col] = df_filled[col].astype(object)
                    df_filled.loc[mask, col] = fill_label
                    applied_value = fill_label

                status = 'Treated'
                treated_count += 1

            # Count missing after treatment
            after_mask, after_missing_cnt, _, _ = self.detector.detect_column(df_filled[col])
            total_missing_after += after_missing_cnt

            resolved_cnt = max(0, missing_cnt_struct - after_missing_cnt)
            resolved_pct = round((resolved_cnt / missing_cnt_struct * 100), 1) if missing_cnt_struct > 0 else 100.0

            audit_entries.append({
                'column': col,
                'detected_type': col_plan.get('detected_type', 'categorical'),
                'strategy': strategy,
                'before': raw_missing_cnt,
                'structural': missing_cnt_struct,
                'after': after_missing_cnt,
                'resolved_pct': resolved_pct,
                'fill_value': str(applied_value) if applied_value is not None else None,
                'indicator_created': indicator_col_name,
                'reason': col_plan.get('reason', ''),
                'status': status
            })

        # Apply column drops
        if dropped_cols:
            df_filled = df_filled.drop(columns=dropped_cols, errors='ignore')

        # Compute summary
        total_resolved = max(0, total_missing_before - total_missing_after)
        overall_resolved_pct = round((total_resolved / total_missing_before * 100), 1) if total_missing_before > 0 else 100.0

        audit_summary = {
            'total_missing_before': total_missing_before,
            'total_missing_after': total_missing_after,
            'total_resolved': total_resolved,
            'overall_resolved_pct': overall_resolved_pct,
            'columns_treated': treated_count,
            'columns_dropped': dropped_cols,
            'indicators_created': created_indicators,
            'missingness': audit_entries
        }

        # Validation Checks
        target_untouched = True
        if target_col and target_col in df_filled:
            target_untouched = bool(df_filled[target_col].isna().sum() == df_structural[target_col].isna().sum())

        id_cols = [c for c, r in roles.items() if r == 'Identifier' and c in df_filled]
        id_untouched = True
        for ic in id_cols:
            if df_filled[ic].isna().sum() != df_structural[ic].isna().sum():
                id_untouched = False

        validation_report = {
            'rows_preserved': bool(len(df_filled) == len(df_structural)),
            'columns_count_initial': int(len(df_structural.columns)),
            'columns_count_filled': int(len(df_filled.columns)),
            'target_untouched': bool(target_untouched),
            'identifiers_untouched': bool(id_untouched),
            'all_treated_resolved': bool(all(e['after'] == 0 for e in audit_entries if e['status'] == 'Treated')),
            'label': 'for reporting/export; models do not use these statistics'
        }

        return df_filled, audit_summary, validation_report
