import React, { useState, useEffect, useCallback } from 'react';
import { Brush, Blocks, Cpu, CheckCircle2, Sparkles, Eye, ShieldCheck, Database, Layers, ArrowDownToLine, Info, AlertTriangle } from 'lucide-react';
import { api, path, message } from '../lib/api';
import { Action, Notice, Stat, Details, Table, Job, Loading, DownloadButton, Tip } from './common';
import { Bars } from './charts';
import { Switch } from './ui/switch';

export default function Preparation({ step, project, refresh }) {
  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [pretrained, setPretrained] = useState(true);
  const [mode, setMode] = useState('quick');
  const [keep, setKeep] = useState([]);

  // Missing-Value Treatment Engine State
  const [plan, setPlan] = useState(null);
  const [strategies, setStrategies] = useState({});
  const [previewData, setPreviewData] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [colSearch, setColSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');

  const name = step === 2 ? 'quality' : step === 4 ? 'features' : 'metrics';

  // Load missing-value plan if on Clean step
  const loadPlan = useCallback(async () => {
    if (step !== 2) return;
    try {
      const res = await api.get(path('/missing-plan'));
      setPlan(res.data);
      // Initialize strategy state from recommendations
      const initial = {};
      res.data.columns.forEach(c => {
        initial[c.column] = {
          strategy: c.recommended_strategy,
          add_indicator: c.add_indicator,
          override_type: c.detected_type
        };
      });
      setStrategies(prev => ({ ...initial, ...prev }));
    } catch (e) {
      // If preceding step not ready, ignore 409
    }
  }, [step]);

  const load = useCallback(async () => {
    try {
      const r = await api.get(path(`/${name}`));
      setResult(r.data);
      if (name === 'quality') {
        setKeep(r.data.columns_dropped.filter(c => !c.dropped).map(c => c.column));
        // If quality already has saved strategies in missingness, sync state
        if (r.data.missingness) {
          const synced = {};
          r.data.missingness.forEach(m => {
            synced[m.column] = {
              strategy: m.strategy || 'none',
              add_indicator: !!m.indicator_created,
              override_type: m.detected_type
            };
          });
          setStrategies(prev => ({ ...synced, ...prev }));
        }
      }
    } catch (e) {
      if (e.response?.status !== 409) setError(message(e));
    }
  }, [name]);

  useEffect(() => {
    load();
    if (step === 2) loadPlan();
    if (project.active_job) {
      api.get(path(`/jobs/${project.active_job}`)).then(r => {
        if (r.data.kind === (step === 2 ? 'clean' : step === 4 ? 'features' : 'train')) {
          setJobId(project.active_job);
          setBusy(['queued', 'running'].includes(r.data.status));
          if (r.data.status === 'failed') setError(r.data.error);
        }
      }).catch(e => setError(message(e)));
    }
  }, [load, loadPlan, project.active_job, step]);

  const done = useCallback(async j => {
    setBusy(false);
    if (j.status === 'failed') {
      setError(j.error);
    } else {
      await load();
      await loadPlan();
      await refresh();
    }
  }, [load, loadPlan, refresh]);

  const run = async () => {
    if (result && !window.confirm('Rerun this step and replace dependent results?')) return;
    setError('');
    setBusy(true);
    setResult(null);
    setPreviewData(null);
    try {
      const r = await api.post(path(step === 2 ? '/clean' : step === 4 ? '/features' : '/train'), {
        keep_columns: keep,
        strategies: strategies,
        pretrained,
        mode
      });
      setJobId(r.data.job_id);
      await refresh();
    } catch (e) {
      setBusy(false);
      setError(message(e));
    }
  };

  const handleStrategyChange = (column, field, value) => {
    setStrategies(prev => ({
      ...prev,
      [column]: {
        ...(prev[column] || {}),
        [field]: value
      }
    }));
  };

  const togglePreview = async () => {
    if (showPreview) {
      setShowPreview(false);
      return;
    }
    setPreviewLoading(true);
    try {
      const res = await api.get(path('/preview-clean'));
      setPreviewData(res.data);
      setShowPreview(true);
    } catch (e) {
      setError(message(e));
    } finally {
      setPreviewLoading(false);
    }
  };

  // Missingness items to display (either from completed quality report or plan)
  const missingItems = result?.missingness || plan?.columns || [];
  const filteredMissing = missingItems.filter(item => {
    const colMatch = (item.column || '').toLowerCase().includes(colSearch.toLowerCase());
    const typeMatch = typeFilter === 'all' || item.detected_type === typeFilter;
    return colMatch && typeMatch;
  });

  return (
    <div>
      {error && <Notice type="error" id="preparation-error">{error}</Notice>}

      <div className="run-configuration">
        <div className="run-icon">
          {step === 2 ? <Brush size={29} /> : step === 4 ? <Blocks size={29} /> : <Cpu size={29} />}
        </div>
        <div className="run-copy">
          <h2>
            {step === 2 ? 'Two-Layer Missing-Value Treatment & Cleaning' : step === 4 ? 'Build the signals that matter' : 'Let your data choose the best model'}
          </h2>
          <p>
            {step === 2
              ? 'Detects tokenized missingness, plans value-based treatments, and produces both leak-safe ML inputs and filled reporting datasets.'
              : step === 4
              ? 'An untouched holdout. Training-only transformations. Reusable features.'
              : 'Training-only cross-validation compares four classifiers against a no-skill baseline.'}
          </p>
        </div>
        <Action id={`run-${step}-button`} disabled={busy} onClick={run}>
          {busy ? 'Processing…' : step === 2 ? 'Apply Treatment & Clean' : step === 4 ? 'Create features' : 'Train models'} <Sparkles size={16} />
        </Action>
      </div>

      {step === 4 && (
        <div className="option-row">
          <div>
            <h3>Pretrained text models</h3>
            <p>Local CPU embeddings · English-first · No hosted inference</p>
          </div>
          <Switch data-testid="pretrained-toggle" checked={pretrained} onCheckedChange={setPretrained} disabled={busy} />
        </div>
      )}

      {step === 5 && (
        <div className="training-options">
          {['quick', 'thorough'].map(m => (
            <button
              data-testid={`training-mode-${m}`}
              className={`mode-option ${mode === m ? 'selected' : ''}`}
              key={m}
              disabled={busy}
              onClick={() => setMode(m)}
            >
              <span className="radio-dot" />
              <div>
                <strong>{m === 'quick' ? 'Quick exploration' : 'Thorough comparison'}</strong>
                <p>{m === 'quick' ? '8 tuning candidates · 3 validation folds' : '25 tuning candidates · 5 validation folds'}</p>
              </div>
              {m === 'quick' && <span className="tag">RECOMMENDED</span>}
            </button>
          ))}
        </div>
      )}

      {jobId && <Job jobId={jobId} onDone={done} />}

      {/* STEP 2: CLEANING & MISSING-VALUE TREATMENT ENGINE */}
      {step === 2 && (
        <>
          {result && (
            <div className="stats-grid four" style={{ marginBottom: 16 }}>
              <Stat label="Rows before" value={result.rows_before?.toLocaleString()} id="rows-before" />
              <Stat label="Rows after" value={result.rows_after?.toLocaleString()} id="rows-after" />
              <Stat
                label="Missing resolved"
                value={`${result.summary_missing?.overall_resolved_pct ?? 100}%`}
                detail={`${result.summary_missing?.total_resolved ?? 0} cells treated`}
                id="missing-resolved-pct"
              />
              <Stat
                label="Categories normalized"
                value={(result.category_inconsistencies_corrected || 0).toLocaleString()}
                id="categories-normalized"
              />
            </div>
          )}

          {/* Dual Download Bar */}
          <div className="option-row" style={{ background: '#FFFFFF', padding: '16px 20px', borderRadius: '12px', border: '1px solid #E6E1D8', margin: '16px 0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h3 style={{ fontSize: '13px', fontWeight: 650, color: '#181B1A', display: 'flex', alignItems: 'center', gap: 6 }}>
                <Database size={15} color="#0E5A44" /> Export & Pipeline Datasets
              </h3>
              <p style={{ fontSize: '11px', color: '#6C7571', margin: '4px 0 0' }}>
                Download value-filled data for reporting or structural data used by the ML pipeline.
              </p>
            </div>
            <div className="button-row" style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <DownloadButton suffix="/download-clean?format=csv&variant=filled" name="leadiq-cleaned-filled.csv" label="Cleaned (Filled) CSV" id="download-filled-csv" />
              <DownloadButton suffix="/download-clean?format=xlsx&variant=filled" name="leadiq-cleaned-filled.xlsx" label="Cleaned (Filled) Excel" id="download-filled-xlsx" />
              <DownloadButton suffix="/download-clean?format=csv&variant=structural" name="leadiq-cleaned-structural.csv" label="Structural CSV" id="download-structural-csv" />
              <DownloadButton suffix="/cleaned?format=csv" name="cleaned.csv" label="Legacy CSV" id="download-cleaned-csv" />
            </div>
          </div>

          {/* Notices */}
          {result?.notices?.map((n, i) => (
            <Notice key={i} id={`clean-notice-${i}`}>{n}</Notice>
          ))}

          {/* Tokens Treated as Missing Banner */}
          <div style={{ background: '#F8F6F1', border: '1px solid #E6E1D8', borderRadius: 10, padding: '12px 18px', margin: '14px 0', fontSize: '11px', color: '#4B534F', display: 'flex', alignItems: 'center', gap: 10 }}>
            <ShieldCheck size={16} color="#0E5A44" />
            <span>
              <strong>Central Token Detector:</strong> Recognized missing representations: <code style={{ background: '#FFFFFF', padding: '2px 5px', borderRadius: 4 }}>empty strings</code>, <code style={{ background: '#FFFFFF', padding: '2px 5px', borderRadius: 4 }}>N/A</code>, <code style={{ background: '#FFFFFF', padding: '2px 5px', borderRadius: 4 }}>NA</code>, <code style={{ background: '#FFFFFF', padding: '2px 5px', borderRadius: 4 }}>null</code>, <code style={{ background: '#FFFFFF', padding: '2px 5px', borderRadius: 4 }}>none</code>, <code style={{ background: '#FFFFFF', padding: '2px 5px', borderRadius: 4 }}>?</code>, <code style={{ background: '#FFFFFF', padding: '2px 5px', borderRadius: 4 }}>-</code>. Numbers like <code>0</code> and explicit <code>"Unknown"</code> are never erased.
            </span>
          </div>

          {/* Missing Values Treatment Table Panel */}
          <div style={{ background: '#FFFFFF', border: '1px solid #E6E1D8', borderRadius: 12, padding: 20, margin: '20px 0' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 12 }}>
              <div>
                <span className="eyebrow" style={{ fontSize: '10px', color: '#0E5A44', fontWeight: 700, letterSpacing: '0.6px' }}>TREATMENT ENGINE</span>
                <h3 style={{ fontSize: '15px', color: '#181B1A', margin: '2px 0 0' }}>Per-Column Strategy & Resolution Matrix</h3>
              </div>
              <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <input
                  type="text"
                  placeholder="Filter columns…"
                  value={colSearch}
                  onChange={e => setColSearch(e.target.value)}
                  style={{ padding: '6px 12px', fontSize: '11px', borderRadius: 6, border: '1px solid #DCD6CD', width: 140 }}
                />
                <select
                  value={typeFilter}
                  onChange={e => setTypeFilter(e.target.value)}
                  style={{ padding: '6px 10px', fontSize: '11px', borderRadius: 6, border: '1px solid #DCD6CD' }}
                >
                  <option value="all">All Types</option>
                  <option value="numeric">Numeric</option>
                  <option value="integer">Integer</option>
                  <option value="categorical_low">Categorical (Low)</option>
                  <option value="categorical_high">Categorical (High)</option>
                  <option value="boolean">Boolean</option>
                  <option value="email">Email</option>
                  <option value="phone">Phone</option>
                  <option value="website">Website</option>
                  <option value="identifier">Identifier</option>
                  <option value="datetime">Datetime</option>
                </select>
                <Action secondary id="preview-clean-button" disabled={previewLoading} onClick={togglePreview}>
                  <Eye size={14} /> {showPreview ? 'Hide Preview' : 'Preview Filled Cells'}
                </Action>
              </div>
            </div>

            {/* Strategy Table */}
            <div className="table-scroll" style={{ maxHeight: 420, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
                <thead>
                  <tr style={{ background: '#FAF8F5', borderBottom: '1px solid #E6E1D8', textAlign: 'left', color: '#5A6460' }}>
                    <th style={{ padding: '10px 12px' }}>Column</th>
                    <th style={{ padding: '10px 12px' }}>Detected Type</th>
                    <th style={{ padding: '10px 12px' }}>Raw Missing</th>
                    <th style={{ padding: '10px 12px' }}>Filled Missing</th>
                    <th style={{ padding: '10px 12px' }}>Resolved %</th>
                    <th style={{ padding: '10px 12px' }}>Strategy</th>
                    <th style={{ padding: '10px 12px' }}>Indicator</th>
                    <th style={{ padding: '10px 12px' }}>Status</th>
                    <th style={{ padding: '10px 12px' }}>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredMissing.map((row, idx) => {
                    const col = row.column;
                    const currentStrat = strategies[col]?.strategy || row.strategy || row.recommended_strategy || 'median';
                    const hasIndicator = strategies[col]?.add_indicator ?? row.add_indicator ?? false;
                    const resolvedPct = row.resolved_pct ?? (row.after === 0 && row.before > 0 ? 100 : row.before === 0 ? 100 : 0);
                    const isTarget = row.is_target || col === project.config?.target;
                    const isId = row.is_identifier || row.detected_type === 'identifier';

                    return (
                      <tr key={col} style={{ borderBottom: '1px solid #F0EDE8', background: idx % 2 === 0 ? '#FFFFFF' : '#FCFBFA' }}>
                        <td style={{ padding: '9px 12px', fontWeight: 600, color: '#181B1A' }}>{col}</td>
                        <td style={{ padding: '9px 12px' }}>
                          <span style={{ fontSize: '10px', background: '#F3EFEA', padding: '2px 6px', borderRadius: 4, color: '#4B534F' }}>
                            {row.detected_type?.replace('_', ' ') || 'categorical'}
                          </span>
                        </td>
                        <td style={{ padding: '9px 12px', color: '#6C7571' }}>
                          {row.before ?? row.missing_count ?? 0}
                          {row.missing_pct ? ` (${row.missing_pct}%)` : ''}
                        </td>
                        <td style={{ padding: '9px 12px', fontWeight: 600, color: (row.after ?? 0) === 0 ? '#0E5A44' : '#9E2A2B' }}>
                          {row.after ?? (currentStrat === 'keep_missing' || isTarget || isId ? (row.before || 0) : 0)}
                        </td>
                        <td style={{ padding: '9px 12px' }}>
                          <span style={{
                            display: 'inline-block',
                            padding: '2px 7px',
                            borderRadius: 10,
                            fontSize: '10px',
                            fontWeight: 600,
                            background: resolvedPct >= 99 ? '#EAF5EE' : resolvedPct > 0 ? '#FEF9EE' : '#FDF2F2',
                            color: resolvedPct >= 99 ? '#0E5A44' : resolvedPct > 0 ? '#9A6A12' : '#9E2A2B'
                          }}>
                            {resolvedPct}%
                          </span>
                        </td>
                        <td style={{ padding: '9px 12px' }}>
                          {isTarget ? (
                            <span style={{ fontSize: '10px', color: '#767E79', fontStyle: 'italic' }}>Excluded from training</span>
                          ) : isId ? (
                            <span style={{ fontSize: '10px', color: '#767E79', fontStyle: 'italic' }}>Preserved ID</span>
                          ) : (
                            <select
                              value={currentStrat}
                              onChange={e => handleStrategyChange(col, 'strategy', e.target.value)}
                              disabled={busy}
                              style={{ padding: '4px 8px', fontSize: '11px', borderRadius: 4, border: '1px solid #DCD6CD', background: '#FFFFFF' }}
                            >
                              <option value="median">Median</option>
                              <option value="mean">Mean</option>
                              <option value="zero">Zero (0)</option>
                              <option value="unknown">Explicit Unknown</option>
                              <option value="most_frequent">Most Frequent (Mode)</option>
                              <option value="group_rare_unknown">Group Rare & Unknown</option>
                              <option value="presence_flag">Presence Flag</option>
                              <option value="knn">KNN Imputer</option>
                              <option value="iterative">Iterative Imputer</option>
                              <option value="drop_column">Drop Column</option>
                              <option value="keep_missing">Leave Missing</option>
                            </select>
                          )}
                        </td>
                        <td style={{ padding: '9px 12px', textAlign: 'center' }}>
                          {!isTarget && !isId ? (
                            <input
                              type="checkbox"
                              checked={hasIndicator}
                              onChange={e => handleStrategyChange(col, 'add_indicator', e.target.checked)}
                              disabled={busy}
                              title="Add missingness indicator column"
                            />
                          ) : '—'}
                        </td>
                        <td style={{ padding: '9px 12px' }}>
                          <span style={{
                            fontSize: '10px',
                            padding: '2px 6px',
                            borderRadius: 4,
                            background: row.status === 'Treated' || resolvedPct >= 99 ? '#EAF5EE' : row.status === 'Dropped' ? '#FDF2F2' : '#F3EFEA',
                            color: row.status === 'Treated' || resolvedPct >= 99 ? '#0E5A44' : row.status === 'Dropped' ? '#9E2A2B' : '#4B534F'
                          }}>
                            {row.status || (resolvedPct >= 99 ? 'Treated' : 'Preserved')}
                          </span>
                        </td>
                        <td style={{ padding: '9px 12px', fontSize: '10px', color: '#767E79', maxWidth: 220 }}>
                          {row.reason}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* PREVIEW MODAL / DRAWER */}
          {showPreview && previewData && (
            <div style={{ background: '#FFFFFF', border: '1px solid #C6E6D2', borderRadius: 12, padding: 20, margin: '16px 0', boxShadow: '0 4px 14px rgba(14,90,68,0.06)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <h3 style={{ fontSize: '14px', fontWeight: 650, color: '#0E5A44', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Eye size={16} /> Treatment Preview: Highlighted Filled Cells (Head 15 Rows)
                </h3>
                <span style={{ fontSize: '11px', color: '#6C7571' }}>
                  {previewData.filled_cells?.length || 0} missing cells filled in sample
                </span>
              </div>
              <p style={{ fontSize: '11px', color: '#4B534F', marginBottom: 14 }}>
                Cells highlighted in soft green indicate values that were previously missing and have now been populated by your chosen strategies.
              </p>
              <div className="table-scroll" style={{ maxHeight: 300, overflowY: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
                  <thead>
                    <tr style={{ background: '#FAF8F5', borderBottom: '1px solid #E6E1D8' }}>
                      {Object.keys(previewData.filled_sample?.[0] || {}).map(c => (
                        <th key={c} style={{ padding: '8px 10px', textAlign: 'left', color: '#5A6460' }}>{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewData.filled_sample?.map((row, rIdx) => (
                      <tr key={rIdx} style={{ borderBottom: '1px solid #F0EDE8' }}>
                        {Object.entries(row).map(([col, val]) => {
                          const isFilled = previewData.filled_cells?.some(fc => fc.row === rIdx && fc.column === col);
                          return (
                            <td
                              key={col}
                              style={{
                                padding: '6px 10px',
                                background: isFilled ? '#EAF5EE' : 'transparent',
                                color: isFilled ? '#0E5A44' : '#181B1A',
                                fontWeight: isFilled ? 650 : 400
                              }}
                              title={isFilled ? 'Filled by treatment strategy' : undefined}
                            >
                              {String(val ?? '—')}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Bar Chart showing post-cleaning missing counts (which now accurately reflects 0 for treated columns!) */}
          {result?.missingness && (
            <Bars
              data={result.missingness.map(c => ({ name: c.column, count: c.after }))}
              title="Missing values after treatment"
              id="clean-missingness"
              insight={
                result.summary_missing?.overall_resolved_pct >= 99
                  ? `All configured missing values successfully treated (${result.summary_missing?.total_resolved} filled). Structural layer preserved unfilled for machine learning models.`
                  : `Treated values resolved. Columns designated as contact or datetime retain intentional gaps.`
              }
            />
          )}

          <Details title="Dropped columns & high-missingness candidates" id="dropped-columns">
            <Table
              id="drop-table"
              rows={result?.columns_dropped || []}
              columns={[
                { key: 'column', label: 'Column' },
                { key: 'reason', label: 'Reason' },
                {
                  key: 'dropped',
                  label: 'Keep',
                  render: (_, r) => (
                    <input
                      data-testid={`keep-${r.column}`}
                      type="checkbox"
                      checked={keep.includes(r.column)}
                      onChange={e => setKeep(e.target.checked ? [...keep, r.column] : keep.filter(c => c !== r.column))}
                    />
                  )
                }
              ]}
            />
          </Details>

          <Details title="Before / after raw sample" id="clean-samples">
            <h3>Raw Upload Sample</h3>
            <Table rows={result?.before_sample} id="before-clean-table" />
            <h3 style={{ marginTop: 16 }}>Treated Cleaned Sample</h3>
            <Table rows={result?.after_sample} id="after-clean-table" />
          </Details>
        </>
      )}

      {/* STEP 4: FEATURES */}
      {result && step === 4 && (
        <>
          <div className="stats-grid four">
            <Stat label="Features created" value={result.count} id="feature-count" />
            <Stat label="Training rows" value={result.split.train.toLocaleString()} detail={`${result.split.train_positives} positives`} id="training-rows" />
            <Stat label="Held-out test rows" value={result.split.test.toLocaleString()} detail={`${result.split.test_positives} positives`} id="test-rows" />
            <Stat label="Class ratio" value={`${result.split.train_negative_positive_ratio.toFixed(1)} : 1`} detail="negative : positive, training" id="class-ratio" />
          </div>
          {result.warnings.map((w, i) => (
            <Notice type="warning" key={i} id={`feature-warning-${i}`}>{w}</Notice>
          ))}
          {result.pretrained_models.length > 0 && (
            <Table
              rows={result.pretrained_models}
              id="pretrained-models"
              columns={[
                { key: 'model', label: 'Local model' },
                { key: 'column', label: 'Source' },
                { key: 'status', label: 'Status' },
                { key: 'seconds', label: 'Seconds' },
                { key: 'truncated_records', label: 'Truncated rows' }
              ]}
            />
          )}
          <div className="feature-groups">
            {Object.entries(result.groups).map(([g, fs]) => (
              <div className="feature-group" key={g}>
                <span className="mini-label">{g.replaceAll('_', ' ')}</span>
                <div>
                  {fs.map(f => (
                    <span className="feature-chip" key={f} data-testid={`feature-${g}-${f}`}>{f}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <Details title="Final transformed feature list" id="final-feature-list">
            <div className="feature-list">
              {result.features.map(f => (
                <code key={f}>{f}</code>
              ))}
            </div>
          </Details>
        </>
      )}

      {/* STEP 5: TRAIN */}
      {result && step === 5 && (
        <>
          <Notice type="success" id="training-success">
            <CheckCircle2 size={17} /> Training complete. {result.best_model} selected by training cross-validated PR-AUC.
          </Notice>
          <div className="stats-grid three">
            <Stat label="Models compared" value={result.models.length} id="model-count" />
            <Stat label="Calibration" value={result.calibration_method} id="calibration-method" />
            <Stat label="Elapsed time" value={`${Math.round(result.total_seconds)}s`} id="training-time" />
          </div>
          {result.warnings.map((w, i) => (
            <Notice type="warning" key={i} id={`training-warning-${i}`}>{w}</Notice>
          ))}
        </>
      )}
    </div>
  );
}