"""Explicit adapters for the reviewed ExpA, Move-IIT and ExpFinal CSV summaries.

No averaging, cohort joining, baseline selection or confidence-interval estimation
is performed. PCK source columns below already contain percentages. Missing rates
and bounds stay NaN; source_flow_hz retains legacy baseline-folder provenance.
Caller IDs label an evaluation; they do not validate or harmonize its protocol.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

from paper_figures import canonical_dataset, canonical_method


METRICS = ("mpjpe_px", "pck_percent", "energy_j_per_sequence",
           "energy_j_per_input_s", "rtf")
_MODEL_RATE = re.compile(r"^(movenet|moveenetofk|eventpointpose)_(5|50)hz$")


def _numeric(frame, name):
    values = pd.to_numeric(frame[name], errors="raise").astype(float)
    if np.isinf(values).any():
        raise ValueError("Infinite values in column: " + name)
    return values


def _rate(frame, hz_columns, period_columns, fallback=None, required=False):
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    candidates = [(name, False) for name in hz_columns if name in frame]
    candidates += [(name, True) for name in period_columns if name in frame]
    for name, is_period in candidates:
        values = _numeric(frame, name)
        if (values.dropna() <= 0).any():
            raise ValueError("Nonpositive rate/period in column: " + name)
        if is_period:
            values = 1.0 / values
        common = result.notna() & values.notna()
        if not np.allclose(result[common], values[common], rtol=1e-6, atol=1e-9):
            raise ValueError("Conflicting rate/period columns: " + name)
        result = result.fillna(values)
    if fallback is not None:
        common = result.notna() & fallback.notna()
        if not np.allclose(result[common], fallback[common], rtol=1e-6, atol=1e-9):
            raise ValueError("Model suffix conflicts with the explicit network rate")
        result = result.fillna(fallback)
    if required and result.isna().any():
        raise ValueError("Every row needs an explicit network rate/period or a known model suffix")
    return result


def _base(path, protocol_id, cohort_id, dataset=None, energy=False):
    if not all(isinstance(v, str) and v.strip() for v in (protocol_id, cohort_id)):
        raise ValueError("Nonempty protocol_id and cohort_id are required")
    path = Path(path).expanduser().resolve()
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Empty summary: " + str(path))
    method_col = "model" if energy and "model" in frame else "method"
    if method_col not in frame:
        raise ValueError("Missing method/model column: " + str(path))
    out = pd.DataFrame(index=frame.index)
    if "dataset" in frame:
        out["dataset"] = frame.dataset.map(canonical_dataset)
        if dataset is not None and not out.dataset.eq(canonical_dataset(dataset)).all():
            raise ValueError("Dataset argument conflicts with the source dataset column")
    elif dataset is not None:
        out["dataset"] = canonical_dataset(dataset)
    else:
        raise ValueError("Dataset missing from summary; provide dataset explicitly")
    for name in ("metric_protocol", "protocol_id", "cohort_id"):
        if name in frame:
            out["source_" + name] = frame[name]
            counts = frame.groupby(out.dataset)[name].nunique(dropna=False)
            if counts.gt(1).any():
                raise ValueError("Mixed source {} within dataset(s): {}; select a source explicitly".format(
                    name, ", ".join(counts.index[counts.gt(1)])))
    methods, suffix_rates = [], []
    for value in frame[method_col]:
        match = _MODEL_RATE.fullmatch(str(value)) if energy else None
        methods.append(canonical_method(match.group(1) if match else value))
        suffix_rates.append(float(match.group(2)) if match else np.nan)
    out["method"] = methods
    out["source_method"] = frame[method_col]
    out["network_hz"] = _rate(frame, ("network_hz", "detection_rate_hz", "net_hz"),
                              ("network_period_s", "net_period_s"),
                              pd.Series(suffix_rates, index=frame.index), required=True)
    out["source_flow_hz"] = _rate(frame, ("flow_hz", "flow_rate_hz"), ("flow_period_s",))
    out["flow_hz"] = out.source_flow_hz.where(out.method.eq(canonical_method("moveenet_ofk")))
    out["output_hz"] = _rate(frame, ("output_hz",), ("output_period_s", "output_period"))
    if "operating_point" in frame:
        if frame.operating_point.isna().any():
            raise ValueError("Missing operating_point in an explicit operating-point column")
        out["operating_point"] = frame.operating_point.astype(str)
    else:
        def label(row):
            fields = ["net={:g}Hz".format(row.network_hz)]
            for name in ("flow_hz", "output_hz"):
                value = row[name]
                if pd.notna(value):
                    fields.append("{}={:g}Hz".format(name.replace("_hz", ""), value))
            return "; ".join(fields)
        out["operating_point"] = out.apply(label, axis=1)
    out["protocol_id"], out["cohort_id"] = protocol_id, cohort_id
    out["source_file"], out["source_row"] = str(path), np.arange(2, len(frame) + 2)
    aggregation = next((name for name in ("aggregation", "accuracy_aggregation") if name in frame), None)
    out["aggregation"] = frame[aggregation].fillna("") if aggregation else ""
    for name in ("accuracy_aggregation", "n_sequences", "n_samples", "n_points",
                 "valid_joints", "valid_mpjpe_joints", "valid_pck_joints",
                 "total_evaluated_joint_points", "repetitions_per_sequence"):
        if name in frame:
            out[name] = frame[name]
    for metric in METRICS:
        for suffix in ("", "_low", "_high"):
            out[metric + suffix] = np.nan
    return frame, out


def _metric(frame, out, target, source, bounds=None, required=False):
    if source not in frame:
        if required:
            raise ValueError("Unsupported summary: missing column " + source)
        if bounds and any(name in frame for name in bounds):
            raise ValueError("Confidence bounds supplied without their estimate: " + source)
        return
    values = _numeric(frame, source)
    if (values.dropna() < 0).any() or (target == "pck_percent" and (values.dropna() > 100).any()):
        raise ValueError("Metric outside its valid range: " + source)
    out[target] = values
    if bounds and any(name in frame for name in bounds):
        if not all(name in frame for name in bounds):
            raise ValueError("Both confidence bounds must be supplied: " + str(bounds))
        lo, hi = (_numeric(frame, name) for name in bounds)
        if (lo.notna() != hi.notna()).any() or (lo > hi).any():
            raise ValueError("Incomplete or reversed confidence bounds: " + str(bounds))
        if (values.isna() & lo.notna()).any():
            raise ValueError("Confidence bounds exist for a missing estimate: " + source)
        out[target + "_low"], out[target + "_high"] = lo, hi


def read_expa_summary(path, dataset, protocol_id, cohort_id):
    """Read dataset-weighted ExpA/Comparison/ExpB summaries; retain every row.

    Requires mpjpe_px and pck_0.2 (percent). Standard deviations and SEM are
    deliberately not interpreted as confidence intervals. Select baseline source
    flow copies explicitly before plotting when the CSV includes duplicate runs.
    """
    frame, out = _base(path, protocol_id, cohort_id, dataset)
    _metric(frame, out, "mpjpe_px", "mpjpe_px", required=True)
    _metric(frame, out, "pck_percent", "pck_0.2", required=True)
    return out


def read_moveiit_accuracy_summary(path, protocol_id, cohort_id):
    """Read the corrected two-method or native-time three-method Move-IIT summary.

    Protocol/cohort IDs must distinguish those evaluations. Rates absent from the
    summary are not reconstructed from notebook defaults or from folder names.
    """
    frame, out = _base(path, protocol_id, cohort_id, dataset="moveiit")
    _metric(frame, out, "mpjpe_px", "mpjpe_px",
            ("mpjpe_px_ci_low", "mpjpe_px_ci_high"), required=True)
    _metric(frame, out, "pck_percent", "pck02_percent",
            ("pck02_percent_ci_low", "pck02_percent_ci_high"), required=True)
    return out


def read_energy_summary(path, protocol_id, cohort_id, dataset=None):
    """Read ExpFinal or Move-IIT energy/runtime and already-paired accuracy CSVs.

    Accepts energy_mean_j; normalized energy and RTF are optional for the paired
    Move-IIT accuracy table. Never joins the full-59 accuracy and 12-recording
    energy cohorts. Only observed model suffixes _5hz/_50hz supply missing rates.
    """
    frame, out = _base(path, protocol_id, cohort_id, dataset, energy=True)
    _metric(frame, out, "energy_j_per_sequence", "energy_mean_j",
            ("energy_ci95_low", "energy_ci95_high"), required=True)
    _metric(frame, out, "energy_j_per_input_s", "energy_j_per_input_second",
            ("normalized_energy_ci95_low", "normalized_energy_ci95_high"))
    _metric(frame, out, "rtf", "real_time_factor", ("rtf_ci95_low", "rtf_ci95_high"))
    if "mpjpe_weighted_px" in frame or "pck_weighted_percent" in frame:
        _metric(frame, out, "mpjpe_px", "mpjpe_weighted_px",
                ("mpjpe_ci95_low", "mpjpe_ci95_high"), required=True)
        _metric(frame, out, "pck_percent", "pck_weighted_percent",
                ("pck_ci95_low", "pck_ci95_high"), required=True)
    elif "mpjpe_px" in frame or "pck02_percent" in frame:
        _metric(frame, out, "mpjpe_px", "mpjpe_px",
                ("mpjpe_px_ci_low", "mpjpe_px_ci_high"), required=True)
        _metric(frame, out, "pck_percent", "pck02_percent",
                ("pck02_percent_ci_low", "pck02_percent_ci_high"), required=True)
    elif not {"real_time_factor", "energy_j_per_input_second"}.issubset(frame.columns):
        raise ValueError("Unsupported energy summary: runtime/normalization or paired accuracy needed")
    return out
