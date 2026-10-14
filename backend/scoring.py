import re
import numpy as np
import pandas as pd
import joblib
from storage import folder,read,save,complete
from features import engineer
from cleaning import structural

def score_job(pid,scope,log,new_df=None,new_meta=None):
    trained = joblib.load(folder(pid) / 'trained.joblib')
    b = trained['bundle']
    cfg = b['config']
    spec = b['spec']
    drift = []

    if new_df is not None:
        missing = [c for c in spec['required'] if c not in new_df]
        if missing:
            raise ValueError('Missing required columns: ' + ', '.join(missing))
        raw = new_df.copy()
        clean, _ = structural(new_df, cfg)
        log(15, 'Applying the saved structural transforms and pretrained features')
        X, _ = engineer(clean, spec, cfg, log, False)
        missing_features = [c for c in b['X'] if c not in X]
        if missing_features:
            raise ValueError('Saved pipeline inputs are missing: ' + ', '.join(missing_features))
        X = X.reindex(columns=b['X'].columns)
        train = b['X'].loc[b['train_ids']]
        for c in X:
            if ' embedding ' in c:
                continue
            if pd.api.types.is_numeric_dtype(X[c]):
                low, high = train[c].min(), train[c].max()
                outside = int(((X[c] < low) | (X[c] > high)).sum())
                if outside:
                    drift.append({'column': c, 'kind': 'Outside training range', 'detail': f'{outside} rows outside [{low:.4g}, {high:.4g}]'})
            else:
                unseen = set(X[c].dropna().astype(str)) - set(train[c].dropna().astype(str))
                if unseen:
                    drift.append({'column': c, 'kind': 'Unseen categories', 'detail': ', '.join(sorted(unseen)[:8])})
            diff = float(X[c].isna().mean() - train[c].isna().mean())
            if abs(diff) > 0.1:
                drift.append({'column': c, 'kind': 'Missingness changed', 'detail': f'{diff:+.1%} versus training'})
        extras = [c for c in raw if c not in spec['required'] and c != cfg['target']]
        if extras:
            drift.append({'column': ', '.join(extras), 'kind': 'Extra columns ignored', 'detail': 'Not required by the saved model.'})
        source = 'new'
        in_sample = np.zeros(len(raw), dtype=bool)
        source_df = clean
    else:
        original = pd.read_pickle(folder(pid) / 'raw.pkl')
        p_filled = folder(pid) / 'clean_filled.pkl'
        p_clean = folder(pid) / 'clean.pkl'
        clean_stored = pd.read_pickle(p_filled) if p_filled.exists() else (pd.read_pickle(p_clean) if p_clean.exists() else original)
        
        if scope == 'test':
            raw = original.loc[b['test_ids']].copy()
            source_df = clean_stored.loc[clean_stored.index.isin(b['test_ids'])].copy() if not clean_stored.empty else raw
            if len(source_df) != len(raw):
                source_df = clean_stored.iloc[:len(raw)].copy()
            X = b['X'].loc[b['test_ids']].copy()
            in_sample = np.zeros(len(raw), dtype=bool)
        else:
            raw = original.copy()
            clean, _ = structural(raw, cfg)
            X, _ = engineer(clean, spec, cfg, log, False)
            X = X.reindex(columns=b['X'].columns)
            in_sample = raw.index.isin(b['train_ids'])
            source_df = clean_stored if len(clean_stored) == len(raw) else clean

        source = scope

    log(65, f'Scoring {len(X)} records with the saved calibrated model')
    model_probs = np.clip(trained['calibrated'].predict_proba(X)[:, 1], 0.0001, 0.9999)
    oof_map = trained.get('oof_predictions', {})
    test_map = trained.get('test_predictions', {})

    raw_indices = list(raw.index)
    n_rows = len(raw)
    prob_arr = np.zeros(n_rows, dtype=float)
    sources = []
    statuses = []

    for i, idx in enumerate(raw_indices):
        idx_key = int(idx) if str(idx).isdigit() else idx
        if scope == 'new':
            prob_arr[i] = model_probs[i]
            sources.append('new_file')
            statuses.append('New unlabeled lead')
        elif scope == 'test':
            if idx_key in test_map:
                prob_arr[i] = test_map[idx_key]
            else:
                prob_arr[i] = model_probs[i]
            sources.append('held_out')
            statuses.append('Held-out test set')
        else:  # scope == 'all'
            if idx_key in oof_map:
                prob_arr[i] = oof_map[idx_key]
                sources.append('out_of_fold')
                statuses.append('Training out-of-fold (unseen fold)')
            elif idx_key in test_map:
                prob_arr[i] = test_map[idx_key]
                sources.append('held_out')
                statuses.append('Held-out test set')
            else:
                prob_arr[i] = model_probs[i]
                sources.append('operational')
                statuses.append('Operational lead')

    probability = np.clip(prob_arr, 0.0001, 0.9999)
    reference = trained['train_reference']
    scores = np.clip(np.searchsorted(reference, probability, side='right') / len(reference) * 100, 0.0, 100.0)
    tiers = np.select([scores >= 90, scores >= 70, scores >= 40], ['Very High', 'High', 'Medium'], default='Low')

    # Detect or preserve canonical lead identifier
    id_col = next((c for c, r in cfg.get('roles', {}).items() if r == 'Identifier' and c in raw), None)
    if not id_col:
        id_col = next((c for c in raw if c.lower() in ['lead_id', 'id', 'leadid', 'customerid', 'customer_id']), None)

    if id_col and id_col in raw:
        lead_ids = raw[id_col].astype(str).tolist()
    else:
        lead_ids = [f'LEAD-{int(i)+1:05d}' for i in raw.index]

    # Detect company / business name column
    name_col = next((c for c, r in cfg.get('roles', {}).items() if r == 'Business Name' and c in raw), None)
    if not name_col:
        name_col = next((c for c in raw if re.search(r'company|business|name', c, re.I)), None)
    company_names = raw[name_col].astype(str).tolist() if (name_col and name_col in raw) else [''] * len(raw)

    # Build output table using cleaned data values
    output = source_df.copy().reset_index(drop=True)
    
    # Ensure canonical fields are present for API routing without clobbering original columns
    if 'lead_id' not in output.columns:
        output['lead_id'] = lead_ids
    else:
        output['lead_id'] = lead_ids
    if name_col and name_col in output and name_col != 'company_name':
        output['company_name'] = company_names
        
    output['predicted_conversion_probability'] = np.round(probability, 4)
    output['lead_score'] = np.round(scores, 2)
    output['priority_tier'] = tiers
    output['prediction_source'] = sources
    output['prediction_status'] = statuses
    # Documented aliases for frontend compatibility
    output['probability'] = output['predicted_conversion_probability']
    output['tier'] = output['priority_tier']
    output['in_sample'] = [s == 'out_of_fold' for s in sources]
    output['row_position'] = np.arange(len(raw))
    output['source_row'] = [int(i) + 2 for i in raw.index]

    # Sort descending by lead score and calibrated probability; deterministic tie-break by row_position
    output = output.sort_values(['lead_score', 'predicted_conversion_probability', 'row_position'], ascending=[False, False, True], kind='stable').reset_index(drop=True)
    output.insert(0, 'rank', np.arange(1, len(output) + 1))
    output.insert(1, 'contact_order', output['rank'])

    # Save outputs
    output.to_pickle(folder(pid) / 'scores.pkl')
    joblib.dump({'X': X, 'raw': raw, 'clean': source_df, 'config': cfg}, folder(pid) / 'scored.joblib')

    # Calculate decile histogram
    histogram = []
    for low in range(0, 100, 10):
        c_mask = (scores >= low) & (scores < (low + 10) if low < 90 else scores <= 100)
        histogram.append({
            'name': f'{low}–{low+10}',
            'count': int(c_mask.sum()),
            'tier': 'Very High' if low >= 90 else 'High' if low >= 70 else 'Medium' if low >= 40 else 'Low'
        })

    # Calculate actual conversion rates per tier if target column is present
    tier_actual_rates = {}
    tier_predicted_rates = {}
    target_col = cfg.get('target')
    if target_col and target_col in raw:
        y_true = (raw[target_col].astype(str).str.strip().str.lower() == cfg.get('positive', 'yes').lower()).astype(int).values
        for t in ['Very High', 'High', 'Medium', 'Low']:
            t_mask = (tiers == t)
            if t_mask.sum() > 0:
                tier_actual_rates[t] = float(y_true[t_mask].mean())
                tier_predicted_rates[t] = float(probability[t_mask].mean())
            else:
                tier_actual_rates[t] = 0.0
                tier_predicted_rates[t] = 0.0

    tier_rates_list = [tier_actual_rates.get(t, 0.0) for t in ['Very High', 'High', 'Medium', 'Low']]
    is_monotonic = bool(tier_rates_list[0] >= tier_rates_list[1] >= tier_rates_list[2] >= tier_rates_list[3])
    separation_gap = float(tier_rates_list[0] - tier_rates_list[3]) if tier_rates_list else 0.0

    summary = {
        'total': len(output),
        'tiers': {t: int((tiers == t).sum()) for t in ['Very High', 'High', 'Medium', 'Low']},
        'tier_actual_rates': tier_actual_rates,
        'tier_predicted_rates': tier_predicted_rates,
        'is_monotonic_separation': is_monotonic,
        'separation_gap': separation_gap,
        'histogram': histogram,
        'source': source,
        'in_sample_count': int((output['prediction_source'] == 'out_of_fold').sum()),
        'drift': drift,
        'columns': list(output.columns),
        'new_file': new_meta
    }
    save(pid, 'score_summary', summary)
    complete(pid, 7)
    log(100, f'Scored {len(output)} leads: calibrated probabilities, percentile scores, and priority tiers saved')

def get_ranked_export_df(scores_df, original_cols=None):
    if original_cols:
        base_cols = [c for c in original_cols if c in scores_df.columns]
    else:
        meta_cols = {
            'rank', 'contact_order', 'predicted_conversion_probability', 'probability',
            'lead_score', 'priority_tier', 'tier', 'prediction_source', 'prediction_status',
            'in_sample', 'row_position', 'source_row'
        }
        base_cols = [c for c in scores_df.columns if c not in meta_cols]
    
    ranked_cols = [
        'rank',
        'predicted_conversion_probability',
        'lead_score',
        'priority_tier',
        'contact_order',
        'prediction_source',
        'prediction_status'
    ]
    export_cols = base_cols + [c for c in ranked_cols if c in scores_df.columns]
    return scores_df[export_cols].copy()

def leads(pid, search='', tier='', page=1, page_size=20, sort='contact_order', descending=False, filter_column='', filter_value='', top_n=None):
    df = pd.read_pickle(folder(pid) / 'scores.pkl')
    if top_n:
        df = df.nsmallest(top_n, 'contact_order' if 'contact_order' in df else 'rank')
    if tier:
        df = df[df['tier'] == tier]
    if search:
        mask = pd.Series(False, index=df.index)
        for c in df:
            mask |= df[c].astype(str).str.contains(search, case=False, regex=False, na=False)
        df = df[mask]
    if filter_column and filter_column in df and filter_value:
        df = df[df[filter_column].astype(str).str.contains(filter_value, case=False, regex=False, na=False)]
    if sort in df:
        try:
            df = df.sort_values(sort, ascending=not descending, kind='stable')
        except TypeError:
            df = df.sort_values(sort, key=lambda s: s.astype(str), ascending=not descending, kind='stable')
    total = len(df)
    s = df.iloc[(page - 1) * page_size:page * page_size]
    return {
        'rows': s.astype(object).where(pd.notna(s), None).to_dict('records'),
        'total': total,
        'page': page,
        'pages': max(1, int(np.ceil(total / page_size))),
        'columns': list(df.columns)
    }