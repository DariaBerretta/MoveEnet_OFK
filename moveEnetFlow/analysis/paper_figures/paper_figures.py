"""Small, data-driven paper plotting API; Python 3.8+, Matplotlib 3.3+.

Inputs are already aggregated, explicit experimental operating points. These
functions check internal consistency; they do not establish scientific validity
or aggregate raw runs. Bounds are absolute endpoints, never inferred error bars.

Axis overrides: {"eh36m": {"mpjpe_px": {"ylim": [0, 30], "xticks": [5, 20, 50]}}}.
Use y-metric keys; for energy/RTF the key is "energy_j_per_input_s".
Render functions return a Figure. export_figure returns format -> absolute path.
"""
import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import is_color_like
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.transforms import Bbox


def _key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def canonical_method(value):
    aliases = {"movenet": "movenet", "moveenet": "movenet",
               "movenetonly": "movenet", "moveenetonly": "movenet",
               "movenetofk": "moveenetofk", "moveenetofk": "moveenetofk",
               "movenetof": "moveenetofk", "moveenetof": "moveenetofk",
               "mofk": "moveenetofk", "yolo": "yolo", "yolopose": "yolo",
               "openpose": "openpose", "op": "openpose",
               "eventpointpose": "eventpointpose", "epp": "eventpointpose"}
    if _key(value) not in aliases:
        raise ValueError("Unknown method {!r}; use a documented canonical ID.".format(value))
    return aliases[_key(value)]


def canonical_dataset(value):
    aliases = {"eh36m": "eh36m", "h36m": "eh36m", "dhp19": "dhp19",
               "moveiit": "moveiit", "moveiithpe": "moveiit"}
    if _key(value) not in aliases:
        raise ValueError("Unknown dataset {!r}; expected eh36m, dhp19 or moveiit.".format(value))
    return aliases[_key(value)]


def _merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_style(path=None):
    """Load the bundled style; an optional JSON file can override any subset."""
    with Path(__file__).with_name("style.json").open(encoding="utf-8") as stream:
        style = json.load(stream)
    if path is not None:
        with Path(path).open(encoding="utf-8") as stream:
            _merge(style, json.load(stream))
    methods = style["methods"]
    for entry in methods.values():
        if not is_color_like(entry["color"]):
            raise ValueError("Invalid method colour: {!r}".format(entry["color"]))
        MarkerStyle(entry["marker"])
    for field in ("color", "marker"):
        if len({entry[field] for entry in methods.values()}) != len(methods):
            raise ValueError("Each method must have a distinct {}.".format(field))
    for order, registry in (("method_order", "methods"), ("dataset_order", "datasets")):
        if len(style[order]) != len(set(style[order])) or set(style[order]) != set(style[registry]):
            raise ValueError("{} must list each {} key exactly once.".format(order, registry))
    return style


def _rc(style):
    fonts, axes = style["fonts"], style["axes"]
    return {"font.family": fonts["family"], "font.size": fonts["label"],
            "mathtext.fontset": fonts["mathtext"], "axes.labelsize": fonts["label"],
            "axes.titlesize": fonts["title"], "xtick.labelsize": fonts["tick"],
            "ytick.labelsize": fonts["tick"], "legend.fontsize": fonts["legend"],
            "axes.linewidth": axes["linewidth"], "xtick.direction": "out",
            "ytick.direction": "out", "xtick.major.size": axes["tick_length"],
            "ytick.major.size": axes["tick_length"], "axes.axisbelow": True,
            "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
            "savefig.bbox": None, "figure.autolayout": False,
            "figure.constrained_layout.use": False}


def _prepare(df, style, datasets, metrics, require_flow=False):
    required = ["dataset", "method", "network_hz", "flow_hz", "output_hz",
                "protocol_id", "cohort_id"] + list(metrics)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: {}".format(", ".join(missing)))
    data = df.copy()
    data["dataset"] = data["dataset"].map(canonical_dataset)
    data["method"] = data["method"].map(canonical_method)
    chosen = ([name for name in style["dataset_order"] if name in set(data.dataset)]
              if datasets is None else [canonical_dataset(d) for d in
                                        ([datasets] if isinstance(datasets, str) else datasets)])
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("Select at least one dataset, with no duplicates.")
    absent = set(chosen) - set(data.dataset)
    if absent:
        raise ValueError("Requested datasets have no rows: {}".format(sorted(absent)))
    data = data[data.dataset.isin(chosen)].copy()
    for column in ("protocol_id", "cohort_id"):
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ValueError("{} must be explicit and non-empty.".format(column))
        for dataset, group in data.groupby("dataset"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError("{} mixes {} values; select one explicitly.".format(dataset, column))
    for column in set(metrics) | {"network_hz", "flow_hz", "output_hz"}:
        data[column] = pd.to_numeric(data[column], errors="raise")
        values = data[column].to_numpy(dtype=float)
        if column in ("flow_hz", "output_hz"):
            if np.isinf(values).any() or (values[np.isfinite(values)] <= 0).any():
                raise ValueError("{} must be positive and finite when known, or absent (NaN).".format(column))
        elif not np.isfinite(values).all():
            raise ValueError("{} contains missing or nonfinite values.".format(column))
        if column == "network_hz" and (values <= 0).any():
            raise ValueError("{} must be positive.".format(column))
        if column in ("mpjpe_px", "energy_j_per_sequence", "energy_j_per_input_s", "rtf") and (values < 0).any():
            raise ValueError("{} must be nonnegative.".format(column))
        if column == "pck_percent" and ((values < 0) | (values > 100)).any():
            raise ValueError("pck_percent must be between 0 and 100, inclusive.")
    flow = data.loc[data.method.eq("moveenetofk"), "flow_hz"]
    if require_flow and flow.isna().any():
        raise ValueError("MoveEnetOFK accuracy-rate curves require known positive flow_hz; select or supply the sourced flow rate explicitly.")
    key = ["dataset", "method", "network_hz", "flow_hz", "output_hz", "protocol_id", "cohort_id"]
    if data.duplicated(key, keep=False).any():
        raise ValueError("Duplicate operating points: method/rates/protocol/cohort must be unique; aggregate or select upstream.")
    for metric in metrics:
        lo, hi = metric + "_low", metric + "_high"
        if (lo in data) != (hi in data):
            raise ValueError("{} requires both _low and _high columns.".format(metric))
        if lo not in data:
            continue
        data[lo], data[hi] = pd.to_numeric(data[lo], errors="raise"), pd.to_numeric(data[hi], errors="raise")
        lows, highs = data[lo].to_numpy(dtype=float), data[hi].to_numpy(dtype=float)
        absent = np.isnan(lows) & np.isnan(highs)
        if ((~absent) & (~np.isfinite(lows) | ~np.isfinite(highs) | (lows > highs))).any():
            raise ValueError("Invalid {} bounds: supply two finite ordered endpoints or two NaNs.".format(metric))
    return data, chosen


def _options(options, dataset, metric):
    normalized = {canonical_dataset(k): v for k, v in (options or {}).items()}
    selected = dict(normalized.get(dataset, {}).get(metric, {}))
    unknown = set(selected) - {"xlim", "ylim", "xscale", "yscale", "xlabel", "ylabel", "xticks", "yticks", "title"}
    if unknown:
        raise ValueError("Unknown axis options: {}".format(sorted(unknown)))
    return selected


def _configure(ax, data, x, y, style, kind, options, default_title):
    for letter, metric in (("x", x), ("y", y)):
        scale = options.get(letter + "scale", style["plots"][kind][letter + "scale"])
        if scale not in ("linear", "log"):
            raise ValueError("Supported axis scales are 'linear' and 'log'.")
        columns = [c for c in (metric, metric + "_low", metric + "_high") if c in data]
        values = data[columns].to_numpy(dtype=float).ravel()
        if scale == "log" and (values[~np.isnan(values)] <= 0).any():
            raise ValueError("{} log axis needs strictly positive values and interval endpoints.".format(metric))
        getattr(ax, "set_" + letter + "scale")(scale)
        limits = options.get(letter + "lim")
        if limits is not None:
            limits = np.asarray(limits, dtype=float)
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError("{}lim must contain two finite increasing values.".format(letter))
            if scale == "log" and (limits <= 0).any():
                raise ValueError("Log axis limits must be strictly positive; limits were not clipped.")
        else:
            # Errorbar collections on logarithmic axes can contaminate artist
            # autoscaling. Derive the view only from original numerical values.
            extent = values[np.isfinite(values)]
            if kind == "energy_rtf" and letter == "x" and style["rtf_reference"]["show"]:
                reference = float(style["rtf_reference"]["value"])
                if not np.isfinite(reference) or (scale == "log" and reference <= 0):
                    raise ValueError("The RTF reference must be finite and positive on a log axis.")
                extent = np.append(extent, reference)
            scaled = np.log10(extent) if scale == "log" else extent
            low, high = float(scaled.min()), float(scaled.max())
            padding = 0.05 * (high - low) if high > low else (0.05 if scale == "log" else 0.05 * max(abs(low), 1))
            limits = np.array([low - padding, high + padding])
            if scale == "log":
                limits = np.power(10.0, limits)
        getattr(ax, "set_" + letter + "lim")(limits)
        getattr(ax, "set_" + letter + "label")(options.get(letter + "label", style["metrics"][metric]))
        if scale == "log":
            axis = getattr(ax, letter + "axis")
            axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=10))
            axis.set_major_formatter(FuncFormatter(lambda value, pos: "{:g}".format(value)))
            axis.set_minor_formatter(NullFormatter())
        ticks = options.get(letter + "ticks")
        if ticks is not None:
            ticks = np.asarray(ticks, dtype=float)
            if ticks.ndim != 1 or not np.isfinite(ticks).all():
                raise ValueError("{}ticks must be a list of finite positions.".format(letter))
            if scale == "log" and (ticks <= 0).any():
                raise ValueError("Log axis tick positions must be strictly positive.")
            getattr(ax, "set_" + letter + "ticks")(ticks)
            if limits is not None:
                getattr(ax, "set_" + letter + "lim")(limits)
    ax.set_title(options.get("title", default_title), pad=5)
    settings = style["axes"]
    for name, spine in ax.spines.items():
        spine.set_visible(name in settings["spines"])
    ax.grid(False, which="minor")
    if settings["grid"]:
        ax.grid(True, which="major", axis=settings["grid_axis"], color=settings["grid_color"],
                linewidth=settings["grid_linewidth"], alpha=settings["grid_alpha"])


def _marker(style, method):
    entry, line = style["methods"][method], style["lines"]
    return dict(color=entry["color"], marker=entry["marker"], markersize=line["marker_size"],
                markeredgewidth=line["marker_edge_width"], markeredgecolor=line["marker_edge_color"])


def _bars(ax, row, x, y, style):
    # Each interval is drawn at its own midpoint, independently of the estimate.
    # Valid intervals need not bracket their associated point estimate.
    line = style["lines"]
    for metric, horizontal in ((x, True), (y, False)):
        low, high = row.get(metric + "_low", np.nan), row.get(metric + "_high", np.nan)
        if pd.isna(low) and pd.isna(high):
            continue
        middle = 0.5 * low + 0.5 * high
        error = np.array([[middle - low], [high - middle]])
        coords = (middle, row[y]) if horizontal else (row[x], middle)
        ax.errorbar(*coords, **({"xerr": error} if horizontal else {"yerr": error}),
                    fmt="none", ecolor=style["methods"][row.method]["color"],
                    elinewidth=line["error_width"], capsize=line["capsize"], zorder=2)


def _scatter(ax, data, x, y, style):
    ax._paper_scatter_points, ax._paper_rate_labels = [], []
    for method in style["method_order"]:
        group = data[data.method.eq(method)].sort_values(["network_hz", "flow_hz", "output_hz"])
        for _, row in group.iterrows():
            _bars(ax, row, x, y, style)
            ax.plot(row[x], row[y], linestyle="none", zorder=3, **_marker(style, method))
            ax._paper_scatter_points.append((row[x], row[y]))
            if len(group) > 1:
                ax._paper_rate_labels.append(((row[x], row[y]), "{:g} Hz".format(row.network_hz),
                                               style["methods"][method]["color"]))


def _place_rate_labels(fig, style):
    """Place short point labels after layout, in display space, without moving data."""
    if not any(getattr(ax, "_paper_rate_labels", []) for ax in fig.axes):
        return
    fig.canvas.draw()
    renderer, px = fig.canvas.get_renderer(), fig.dpi / 72
    offsets = [(8, 8), (-8, 8), (8, -8), (-8, -8), (14, 0), (-14, 0),
               (0, 14), (0, -14), (8, 24), (-8, 24), (8, -24), (-8, -24)]
    for ax in fig.axes:
        pending = getattr(ax, "_paper_rate_labels", [])
        if not pending:
            continue
        frame = ax.get_window_extent(renderer)
        inside = Bbox.from_extents(frame.x0 + 2 * px, frame.y0 + 2 * px,
                                  frame.x1 - 2 * px, frame.y1 - 2 * px)
        radius = (style["lines"]["marker_size"] / 2 + 1) * px
        point_pixels = ax.transData.transform(ax._paper_scatter_points)
        obstacles = [Bbox.from_extents(x - radius, y - radius, x + radius, y + radius)
                     for x, y in point_pixels]
        def edge_distance(item):
            x, y = ax.transData.transform(item[0])
            return min(x - frame.x0, frame.x1 - x, y - frame.y0, frame.y1 - y)
        for xy, label, color in sorted(pending, key=edge_distance):
            if not frame.contains(*ax.transData.transform(xy)):
                continue
            annotation = ax.annotate(label, xy, xytext=(8, 8), textcoords="offset points",
                                     fontsize=style["fonts"]["annotation"], zorder=4,
                                     arrowprops=dict(arrowstyle="-", color=color, linewidth=0.55,
                                                     shrinkA=2, shrinkB=style["lines"]["marker_size"] / 2 + 0.5))
            annotation.set_in_layout(False)
            for dx, dy in offsets:
                annotation.set_position((dx, dy))
                annotation.set_ha("left" if dx > 0 else "right" if dx < 0 else "center")
                annotation.set_va("bottom" if dy > 0 else "top" if dy < 0 else "center")
                annotation.update_positions(renderer)
                box = Text.get_window_extent(annotation, renderer).padded(px)
                if (inside.contains(box.x0, box.y0) and inside.contains(box.x1, box.y1)
                        and not any(box.overlaps(other) for other in obstacles)):
                    obstacles.append(box)
                    break
            else:
                annotation.remove()
                raise ValueError("Rate labels do not fit inside a panel; increase figure width/row height or select fewer operating points.")
    fig.canvas.draw()


def _finish(fig, data, style, title, reserve_mm, flows=None):
    settings, line = style["legend"], style["lines"]
    handles = [Line2D([], [], label=style["methods"][m]["label"], linestyle="none", **_marker(style, m))
               for m in style["method_order"] if m in set(data.method)]
    height_mm = fig.get_figheight() * 25.4
    kwargs = dict(loc=settings["location"], frameon=settings["frame"],
                  columnspacing=settings["column_spacing"], handlelength=settings["handle_length"],
                  labelspacing=settings["label_spacing"])
    columns = min(int(settings["columns"]), len(handles))
    fig.legend(handles=handles, ncol=columns,
               bbox_to_anchor=(0.5, settings["bottom_mm"] / height_mm), **kwargs)
    if flows and len(flows) > 1:
        flow_handles = [Line2D([], [], color="#444444", linestyle=pattern, linewidth=line["width"],
                               label="OF {:g} Hz".format(level)) for level, pattern in flows.items()]
        legend_rows = int(np.ceil(len(handles) / columns))
        flow_y = settings["bottom_mm"] + legend_rows * style["fonts"]["legend"] * 0.52 + 2
        fig.legend(handles=flow_handles, ncol=min(4, len(flow_handles)),
                   bbox_to_anchor=(0.5, flow_y / height_mm), **kwargs)
    if title:
        fig.suptitle(title, fontsize=style["fonts"]["title"] + 1)
    figure = style["figure"]
    fig.tight_layout(rect=(0, reserve_mm / height_mm, 1, 1 - (7 / height_mm if title else 0)),
                     pad=figure["layout_pad"], w_pad=figure["horizontal_pad"], h_pad=figure["vertical_pad"])
    _place_rate_labels(fig, style)
    return fig


def _pair_figure(data, datasets, style, title, kind, x, axis_options, metrics=None):
    metrics = tuple(metrics or ("mpjpe_px", "pck_percent"))
    if not metrics or len(set(metrics)) != len(metrics) or not set(metrics).issubset({"mpjpe_px", "pck_percent"}):
        raise ValueError("metrics must be mpjpe_px and/or pck_percent, without duplicates")
    reserve = style["figure"]["legend_height_mm"]
    levels = sorted(data.loc[data.method.eq("moveenetofk"), "flow_hz"].unique()) if kind == "accuracy_rate" else []
    patterns = style["lines"]["flow_styles"]
    if len(levels) > len(patterns):
        raise ValueError("More flow rates than distinct configured flow_styles; add styles or select rates.")
    flows = dict(zip(levels, patterns))
    reserve += 8 * int(len(flows) > 1)
    height = len(datasets) * style["figure"]["row_height_mm"] + reserve + (7 if title else 0)
    with plt.style.context("default"), plt.rc_context(_rc(style)):
        fig, axes = plt.subplots(len(datasets), len(metrics), squeeze=False,
                                 figsize=((style["figure"]["width_mm"] if len(metrics) == 2
                                           else style["figure"].get("single_width_mm", 105)) / 25.4,
                                          height / 25.4))
        for index, dataset in enumerate(datasets):
            group = data[data.dataset.eq(dataset)]
            row_options = []
            for col, metric in enumerate(metrics):
                ax = axes[index, col]
                options = _options(axis_options, dataset, metric)
                row_options.append(options)
                name = "({}) {}".format(chr(97 + index * 2 + col), style["datasets"][dataset]["label"])
                _configure(ax, group, x, metric, style, kind, options, name)
                if kind == "energy_accuracy":
                    _scatter(ax, group, x, metric, style)
                    continue
                for method in style["method_order"]:
                    subset = group[group.method.eq(method)]
                    method_flows = sorted(subset.flow_hz.unique()) if method == "moveenetofk" else [None]
                    for flow in method_flows:
                        series = subset if flow is None else subset[subset.flow_hz.eq(flow)]
                        series = series.sort_values("network_hz")
                        if series.empty:
                            continue
                        if series.network_hz.duplicated().any() or series.output_hz.nunique(dropna=False) != 1:
                            raise ValueError("A rate curve needs one point per network rate and consistent output_hz (one known value or all missing): {} / {}.".format(dataset, method))
                        ax.plot(series[x], series[metric], linewidth=style["lines"]["width"],
                                linestyle=flows[flow] if flow is not None else "-", **_marker(style, method))
                        for _, row in series.iterrows():
                            _bars(ax, row, x, metric, style)
                if group.network_hz.nunique() <= 12 and options.get("xticks") is None:
                    ax.set_xticks(sorted(group.network_hz.unique()))
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: "{:g}".format(value)))
                    if options.get("xlim") is not None:
                        ax.set_xlim(options["xlim"])
            if kind == "energy_accuracy":
                scales = {o.get("xscale", style["plots"][kind]["xscale"]) for o in row_options}
                limits = [tuple(o["xlim"]) for o in row_options if o.get("xlim") is not None]
                if len(scales) != 1 or (len(limits) == 2 and limits[0] != limits[1]):
                    raise ValueError("Energy panels for a dataset must use the same xscale and xlim.")
                common = limits[0] if limits else (min(a.get_xlim()[0] for a in axes[index]), max(a.get_xlim()[1] for a in axes[index]))
                for ax in axes[index]:
                    ax.set_xlim(common)
        return _finish(fig, data, style, title, reserve, flows)


def render_energy_accuracy(df, style, datasets=None, energy_metric="energy_j_per_sequence", title=None, axis_options=None, metrics=None):
    """Dataset rows; MPJPE left, PCK right; one shared method legend."""
    if energy_metric not in ("energy_j_per_sequence", "energy_j_per_input_s"):
        raise ValueError("Choose energy_j_per_sequence or energy_j_per_input_s explicitly.")
    data, chosen = _prepare(df, style, datasets, [energy_metric, "mpjpe_px", "pck_percent"])
    return _pair_figure(data, chosen, style, title, "energy_accuracy", energy_metric, axis_options, metrics=metrics)


def render_accuracy_rate(df, style, datasets=None, title=None, axis_options=None, metrics=None):
    """Observed network rates; additional flow rates use an independent line key."""
    data, chosen = _prepare(df, style, datasets, ["network_hz", "mpjpe_px", "pck_percent"], require_flow=True)
    return _pair_figure(data, chosen, style, title, "accuracy_rate", "network_hz", axis_options, metrics=metrics)


def render_energy_rtf(df, style, datasets=None, title=None, axis_options=None):
    """One panel per dataset; energy divided by input duration against RTF."""
    x, y = "rtf", "energy_j_per_input_s"
    data, chosen = _prepare(df, style, datasets, [x, y])
    with plt.style.context("default"), plt.rc_context(_rc(style)):
        fig, axes = plt.subplots(1, len(chosen), squeeze=False,
                                 figsize=(style["figure"]["width_mm"] / 25.4, style["figure"]["rtf_height_mm"] / 25.4))
        for index, dataset in enumerate(chosen):
            ax, group = axes[0, index], data[data.dataset.eq(dataset)]
            options = _options(axis_options, dataset, y)
            name = "({}) {}".format(chr(97 + index), style["datasets"][dataset]["label"])
            _configure(ax, group, x, y, style, "energy_rtf", options, name)
            _scatter(ax, group, x, y, style)
            reference = style["rtf_reference"]
            if reference["show"]:
                if ax.get_xscale() == "log" and reference["value"] <= 0:
                    raise ValueError("The RTF reference must be positive on a log axis.")
                ax.axvline(reference["value"], color=reference["color"], linestyle=reference["linestyle"], linewidth=reference["linewidth"], zorder=1)
        return _finish(fig, data, style, title, style["figure"]["legend_height_mm"])


def export_figure(fig, stem, style, metadata=None):
    """Write configured PDF/SVG/PNG and .meta.json, retaining physical size."""
    stem = Path(stem).expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    with plt.style.context("default"), plt.rc_context(_rc(style)):
        for extension in style["export"]["formats"]:
            if extension not in ("pdf", "svg", "png"):
                raise ValueError("Unsupported export format: {}".format(extension))
            path = Path(str(stem) + "." + extension)
            fig.savefig(str(path), format=extension, dpi=style["export"]["dpi"],
                        transparent=style["export"]["transparent"], bbox_inches=None)
            outputs[extension] = str(path)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "figure_size_mm": [float(v) * 25.4 for v in fig.get_size_inches()],
              "style": style, "provenance": metadata or {}}
    path = Path(str(stem) + ".meta.json")
    with path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    outputs["metadata"] = str(path)
    return outputs
