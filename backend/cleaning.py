import re
import numpy as np
import pandas as pd
import phonenumbers
from rapidfuzz import fuzz, process
from storage import folder,read,save,complete,invalidate
from dataio import MISSING,number,records,profile
from missing_engine import MissingValueEngine, CentralMissingDetector

def build_category_variant_mappings(series):
    """
    Finds casing and separator variants within a low-cardinality categorical series.
    Key = casefold + whitespace/separator normalization (spaces, underscores, hyphens).
    Canonical value = most frequent original spelling (tie-break: earliest first seen).
    A value with no variants is NEVER altered.
    Returns: (mapping: dict[str, str], log_details: list[dict])
    """
    valid = series.dropna().astype(str).map(lambda v: v.strip())
    valid = valid[~valid.str.lower().isin(MISSING) & (valid != '')]
    if len(valid) == 0:
        return {}, []
    
    counts = valid.value_counts()
    first_seen = {}
    for idx, val in enumerate(valid):
        if val not in first_seen:
            first_seen[val] = idx
            
    def make_key(s):
        return re.sub(r'[\s_\-]+', ' ', s.casefold()).strip()
        
    groups = {}
    for val in counts.index:
        k = make_key(val)
        groups.setdefault(k, []).append(val)
        
    mapping = {}
    log_details = []
    for k, val_list in groups.items():
        if len(val_list) > 1:
            canonical = max(val_list, key=lambda v: (counts[v], -first_seen[v]))
            variants_mapped = {}
            for v in val_list:
                if v != canonical:
                    mapping[v] = canonical
                    variants_mapped[v] = int(counts[v])
            if variants_mapped:
                log_details.append({
                    'canonical': canonical,
                    'variants': variants_mapped,
                    'total_changed': sum(variants_mapped.values())
                })
                
    return mapping, log_details

def structural(df, config, return_stats=False):
    df = df.copy()
    invalid = {'email': 0, 'phone': 0}
    cat_corrections = 0
    all_cat_logs = []
    config_mappings = config.get('category_mappings', {})

    for col in df:
        role = config.get('roles', {}).get(col, 'Automatic')
        typ = config.get('types', {}).get(col)
        
        # String and Categorical standardization
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            # 1. Standardize missing indicators
            df[col] = df[col].map(lambda v: np.nan if pd.isna(v) or str(v).strip().lower() in MISSING else v)
            
            # 2. Trim and collapse whitespace for all string columns (NEVER blanket title-case!)
            if role not in ['Identifier', 'Email', 'Phone', 'Website'] and typ != 'identifier':
                df[col] = df[col].map(lambda v: re.sub(r'\s+', ' ', str(v).strip()) if pd.notna(v) else np.nan)
            
            # 3. Category variant normalization for low-cardinality categorical columns
            is_excluded_from_cat = role in ['Identifier', 'Email', 'Phone', 'Website', 'Free Text', 'Lead Notes'] or typ in ['identifier', 'free_text']
            is_cat = (role in ['Categorical', 'Category', 'Industry', 'Lead Source', 'Country', 'City', 'State'] or typ == 'categorical')
            if not is_cat and not is_excluded_from_cat and col != config.get('target'):
                if df[col].nunique(dropna=True) <= 100:
                    is_cat = True
            if is_cat:
                col_map = config_mappings.get(col, {})
                col_logs = []
                if not col_map and df[col].nunique(dropna=True) <= 100:
                    col_map, col_logs = build_category_variant_mappings(df[col])
                    if col_logs:
                        all_cat_logs.append({'column': col, 'mappings': col_logs})
                
                if col_map:
                    def clean_cat(v):
                        nonlocal cat_corrections
                        if pd.isna(v):
                            return np.nan
                        s = str(v).strip()
                        if s in col_map:
                            cat_corrections += 1
                            return col_map[s]
                        return s
                    df[col] = df[col].map(clean_cat)

        if col == config.get('target'):
            continue

        if typ == 'numeric' or role in ['Numeric', 'Rating', 'Review Count', 'Business Age']:
            df[col] = df[col].map(number)
            if role == 'Rating':
                df.loc[(df[col] < 0) | (df[col] > 5), col] = np.nan
            if role in ['Review Count', 'Business Age']:
                df.loc[df[col] < 0, col] = np.nan

        # Preserve Yes/No as Yes/No in business dataset; never convert to 1/0 here!
        is_bool = (typ == 'boolean' or role == 'Boolean')
        if not is_bool and (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            vals = set(df[col].dropna().astype(str).str.strip().str.lower().unique())
            if vals and vals.issubset({'yes', 'no', 'true', 'false', 'y', 'n'}):
                is_bool = True

        if is_bool:
            def clean_bool(v):
                if pd.isna(v): return np.nan
                s = str(v).strip().lower()
                if s in ['yes', 'true', 'y', '1']: return 'Yes'
                if s in ['no', 'false', 'n', '0']: return 'No'
                return v
            df[col] = df[col].map(clean_bool)

        if typ == 'datetime' or role == 'Datetime':
            df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed', utc=True).map(lambda v: v.isoformat() if pd.notna(v) else np.nan)

        if role == 'Email':
            valid = df[col].fillna('').astype(str).str.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$')
            invalid['email'] += int((df[col].notna() & ~valid).sum())
            df[f'{col} is_valid'] = valid.astype(int)

        if role == 'Phone':
            def phone(v):
                if pd.isna(v): return None
                try:
                    p = phonenumbers.parse(str(v), 'IN')
                    return phonenumbers.format_number(p, phonenumbers.PhoneNumberFormat.E164) if phonenumbers.is_valid_number(p) else None
                except Exception: return None
            normalized = df[col].map(phone)
            valid = normalized.notna()
            invalid['phone'] += int((df[col].notna() & ~valid).sum())
            df[f'{col} is_valid'] = valid.astype(int)
            df.loc[valid, col] = normalized[valid]

        if role == 'Website':
            df[col] = df[col].map(lambda v: np.nan if pd.isna(v) else (str(v) if re.match(r'^https?://', str(v)) else 'https://' + str(v)))

    if return_stats:
        return df, invalid, cat_corrections, all_cat_logs
    return df, invalid

def get_missing_plan(pid):
    config = read(pid, 'config')
    raw = pd.read_pickle(folder(pid) / 'raw.pkl')
    engine = MissingValueEngine()
    return engine.plan(raw, config)

def clean_job(pid, overrides=None, log=None, strategies=None, custom_tokens=None, keep_columns=None):
    if overrides is None and keep_columns is not None:
        overrides = keep_columns
    if log is None:
        log = lambda *args: None
    config = read(pid, 'config')
    raw = pd.read_pickle(folder(pid) / 'raw.pkl')
    log(8, 'Standardizing missing values, category representations, dates, and numbers')
    
    keep_cols = set(overrides) if isinstance(overrides, list) else set()
    user_strategies = strategies if strategies is not None else (overrides if isinstance(overrides, dict) else {})
    
    df, invalid, cat_corrections, all_cat_logs = structural(raw, config, return_stats=True)
    before = len(df)
    
    # 1. Exact duplicates
    exact = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log(25, f'Removed {exact} exact duplicates; validating identifiers and contacts')
    
    # 2. Identifier duplicate check
    id_cols = [c for c, r in config.get('roles', {}).items() if r == 'Identifier' and c in df]
    if not id_cols and 'lead_id' in df:
        id_cols = ['lead_id']
    id_duplicates_count = int(df[id_cols[0]].duplicated().sum()) if id_cols else 0
    
    # 3. Fuzzy duplicate resolution only for matching names + phone/email/city
    names = [c for c, r in config.get('roles', {}).items() if r == 'Business Name' and c in df]
    cities = [c for c, r in config.get('roles', {}).items() if r == 'City' and c in df]
    phones = [c for c, r in config.get('roles', {}).items() if r == 'Phone' and c in df]
    drop = set()
    conflicts = 0
    
    if names and phones:
        name = names[0]
        phone_col = phones[0]
        phone_groups = df.groupby(phone_col)
        for _, group in phone_groups:
            if len(group) > 1:
                first_idx = group.index[0]
                for other_idx in group.index[1:]:
                    sim = fuzz.ratio(str(df.loc[first_idx, name]), str(df.loc[other_idx, name]))
                    if sim > 92:
                        if str(df.loc[first_idx, config['target']]) != str(df.loc[other_idx, config['target']]):
                            conflicts += 1
                        drop.add(other_idx)
    
    df = df.drop(index=list(drop))
    log(50, f'Removed {len(drop)} conflicting/duplicate records; auditing columns')
    
    # 4. Column auditing (>60% missing or constant or explicit drop)
    candidates = []
    for c in df:
        if c == config['target']:
            continue
        missing = float(df[c].isna().mean())
        constant = df[c].nunique(dropna=True) <= 1
        strat_info = user_strategies.get(c, {})
        strat_name = strat_info.get('strategy') if isinstance(strat_info, dict) else (strat_info if isinstance(strat_info, str) else None)
        
        is_dropped = (c not in keep_cols) and (strat_name == 'drop_column' or (strat_name is None and (missing > 0.6 or constant)))
        if missing > 0.6 or constant or strat_name == 'drop_column':
            candidates.append({
                'column': c,
                'reason': f'{missing:.1%} missing' if missing > 0.6 else ('Constant column' if constant else 'Explicitly dropped'),
                'dropped': is_dropped
            })
            
    drops = [x['column'] for x in candidates if x['dropped']]
    
    excluded_reasons = []
    if exact > 0:
        excluded_reasons.append(f'{exact} exact duplicate rows removed')
    if len(drop) > 0:
        excluded_reasons.append(f'{len(drop)} duplicate contact records resolved')
    if drops:
        excluded_reasons.append(f'Columns dropped: {", ".join(drops)}')
        
    if len(df) < 200:
        raise ValueError(f'Only {len(df)} unique rows remain after cleaning; at least 200 are needed.')
        
    df_structural = df.drop(columns=drops)
    
    log(70, 'Executing Missing-Value Treatment Engine across detected types')
    engine = MissingValueEngine(custom_tokens=custom_tokens)
    df_filled, audit_summary, validation_report = engine.treat(
        df_structural=df_structural,
        user_strategies=user_strategies,
        config=config,
        raw_df=raw
    )
    
    # Target missing check
    target_col = config.get('target')
    notices = ['Conflicting duplicate outcomes were resolved by retaining the most complete record.'] if conflicts else []
    if target_col and target_col in raw:
        target_missing_count = int(raw[target_col].isna().sum())
        if target_missing_count > 0:
            notices.append(
                f"Target outcome column '{target_col}' has {target_missing_count} missing values. "
                f"In accordance with leak-safety rules, these rows are excluded from model training but preserved for operational scoring."
            )
            excluded_reasons.append(f"{target_missing_count} rows with missing target labels excluded from supervised training")

    report = dict(
        rows_before=before,
        rows_after=len(df_structural),
        columns_before=len(raw.columns),
        columns_after=len(df_structural.columns),
        exact_duplicates=exact,
        fuzzy_duplicates=len(drop),
        conflicting_duplicates=conflicts,
        identifier_duplicates=id_duplicates_count,
        category_inconsistencies_corrected=cat_corrections,
        invalid_emails=invalid['email'],
        invalid_phones=invalid['phone'],
        columns_dropped=candidates,
        missingness=audit_summary['missingness'],
        summary_missing=audit_summary,
        validation=validation_report,
        tokens_treated_as_missing=list(engine.detector.tokens),
        excluded_rows_reasons=excluded_reasons,
        before_sample=records(raw),
        after_sample=records(df_filled),
        structural_sample=records(df_structural),
        notices=notices
    )
    
    # Two-layer persistence:
    # Layer 1: structural for features/ML pipeline, filled for viewing/download/reports
    df_structural.to_pickle(folder(pid) / 'clean_structural.pkl')
    df_structural.to_pickle(folder(pid) / 'clean.pkl')
    df_structural.to_pickle(folder(pid) / 'cleaned_business.pkl')
    df_filled.to_pickle(folder(pid) / 'clean_filled.pkl')
    
    strategy_spec = {
        'timestamp': pd.Timestamp.now(tz='UTC').isoformat(),
        'user_strategies': user_strategies,
        'custom_tokens': custom_tokens,
        'dropped_columns': drops,
        'summary': audit_summary
    }
    save(pid, 'cleaning_strategy', strategy_spec)
    save(pid, 'quality', report)
    complete(pid, 2)
    log(100, f'Cleaning complete: {audit_summary["total_resolved"]} missing values treated ({audit_summary["overall_resolved_pct"]}% resolved); datasets saved')

def eda(pid):
    df = pd.read_pickle(folder(pid) / 'clean.pkl')
    cfg = read(pid, 'config')
    target = cfg['target']
    valid = df[target].notna()
    y = (df.loc[valid, target].astype(str).str.lower() == cfg['positive']).astype(int)
    data = df.loc[valid].copy()
    cats = []
    nums = []
    
    for p in profile(data.drop(columns=[target])):
        c = p['name']
        if p['type'] in ['categorical', 'boolean'] and len(cats) < 4:
            temp = pd.DataFrame({'category': data[c].fillna('Missing').astype(str), 'y': y})
            g = temp.groupby('category')['y'].agg(['mean', 'count']).sort_values('count', ascending=False).head(8)
            vals = [{'name': str(k), 'rate': float(v['mean']), 'count': int(v['count'])} for k, v in g.iterrows()]
            cats.append({
                'column': c,
                'data': vals,
                'insight': f'{vals[0]["name"]} is the largest group ({vals[0]["count"]} rows), with a {vals[0]["rate"]:.1%} positive outcome rate.' if vals else ''
            })
        if p['type'] == 'numeric' and len(nums) < 4:
            s = data[c].map(number).dropna()
            if len(s) and s.nunique() > 1:
                logscale = bool(s.skew() > 2 and s.min() >= 0)
                v = np.log1p(s) if logscale else s
                counts, edges = np.histogram(v, bins=12)
                nums.append({
                    'column': c,
                    'log_scale': logscale,
                    'data': [{'name': f'{edges[i]:.1f}–{edges[i+1]:.1f}', 'count': int(n)} for i, n in enumerate(counts)],
                    'insight': f'Median {s.median():.2f}; {len(s)} nonmissing observations.' + (' Axis shows log(1 + value) because the distribution is skewed.' if logscale else '')
                })
                
    numeric = data.select_dtypes(include='number').iloc[:, :12].copy()
    numeric['Target outcome'] = y
    corr = numeric.corr().fillna(0)
    
    result = {
        'balance': [{'name': 'Positive', 'count': int(y.sum())}, {'name': 'Negative', 'count': int((1 - y).sum())}],
        'prevalence': float(y.mean()),
        'categorical': cats,
        'numeric': nums,
        'missingness': [{'name': c, 'count': int(data[c].isna().sum())} for c in data],
        'correlation': {'columns': list(corr.columns), 'values': corr.values.tolist()}
    }
    save(pid, 'eda', result)
    complete(pid, 3)
    return result