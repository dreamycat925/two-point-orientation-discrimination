from __future__ import annotations

import copy
import html
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Two-Point Orientation Discrimination"

DOME_LEVELS_MM: List[float] = [
    0.35,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    6.0,
    8.0,
    10.0,
    12.0,
]

PRACTICE_LEVELS_MM: List[float] = [8.0, 10.0, 12.0]
PRACTICE_PASS_STREAK = 5
PRACTICE_ERRORS_TO_STEP = 2
PRACTICE_SCHEDULE_LEN = 1000

TEST_START_MM = 8.0
TEST_REVERSAL_TARGET = 10
TEST_THRESHOLD_LAST_N = 6
TEST_MAX_TRIALS = 100
TEST_FLOOR_PASS_STREAK = 4
TEST_CEIL_FAIL_STREAK = 2

ORIENTATION_MAP = {1: "縦", 2: "横"}
ORIENTATION_REVERSE_MAP = {"縦": 1, "横": 2}

HAND_OPTIONS = ["右手", "左手"]
FINGER_OPTIONS = ["母指", "示指", "中指", "環指", "小指"]

PHASE_LABELS = {
    "practice": "練習",
    "test": "本番",
    "post": "事後",
}

PHASE_REASON_TEXT = {
    "practice": {
        "pass": "PASS（5問連続正答）",
        "fail": "FAIL（12 mm で2問誤答）",
        "manual_stop": "手動終了",
    },
    "post": {
        "pass": "PASS（5問連続正答）",
        "fail": "FAIL（12 mm で2問誤答）",
        "manual_stop": "手動終了",
    },
    "test": {
        "floor_correct_pass": "PASS（0.35 mm で4問連続正答）",
        "ceil_incorrect_fail": "FAIL（12 mm で2問連続誤答）",
        "reversal_target": "収束完了（10 reversals 到達）",
        "max_trials_nonconvergent": "収束不良（100 trial 到達）",
        "manual_stop": "手動終了",
    },
}

SERIES_1_TEXT = """
2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2,
2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2,
1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2,
1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1,
2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2
"""

SERIES_2_TEXT = """
2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1,
2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1,
2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2,
2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1,
2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2
"""


def parse_fixed_series(text: str) -> List[int]:
    raw = text.replace("\n", " ")
    parts = [part.strip() for part in raw.split(",")]
    out: List[int] = []
    for part in parts:
        if not part:
            continue
        if part not in {"1", "2"}:
            raise ValueError(f"固定系列に不正な値があります: {part}")
        out.append(int(part))
    return out


TEST_SERIES_1 = parse_fixed_series(SERIES_1_TEXT)
TEST_SERIES_2 = parse_fixed_series(SERIES_2_TEXT)
assert len(TEST_SERIES_1) == 100
assert len(TEST_SERIES_2) == 100


def format_mm(x: float) -> str:
    if abs(float(x) - round(float(x))) < 1e-9 and float(x) >= 1.0:
        return str(int(round(float(x))))
    return f"{float(x):.2f}".rstrip("0").rstrip(".")


def levels_text(levels: Sequence[float]) -> str:
    return ", ".join(format_mm(v) for v in levels)


def orientation_label(code: int) -> str:
    return ORIENTATION_MAP[int(code)]


def orientation_code(label: str) -> int:
    return int(ORIENTATION_REVERSE_MAP[str(label)])


def choose_seed(raw: str, salt: int = 0) -> int:
    text = str(raw).strip()
    if text:
        seed = int(text)
        if seed < 0:
            raise ValueError("seed は 0 以上で指定してください。")
        return seed
    return int((time.time_ns() + int(salt)) % (2**31 - 1))


def generate_random_series(n: int, seed: int) -> List[int]:
    rng = np.random.default_rng(int(seed))
    return [int(x) for x in rng.integers(1, 3, size=int(n))]


def nearest_level(levels: Sequence[float], x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    arr = list(levels)
    if not arr:
        return None
    return float(min(arr, key=lambda v: (abs(v - float(x)), v)))


def format_schedule_codes(seq: Sequence[int], per_line: int = 25) -> str:
    lines: List[str] = []
    for i in range(0, len(seq), per_line):
        lines.append(", ".join(str(int(v)) for v in seq[i : i + per_line]))
    return "\n".join(lines)


def format_schedule_labels(seq: Sequence[int], per_line: int = 25) -> str:
    labels = [orientation_label(int(v)) for v in seq]
    lines: List[str] = []
    for i in range(0, len(labels), per_line):
        lines.append(" ".join(labels[i : i + per_line]))
    return "\n".join(lines)


def compute_test_threshold_summary(reversal_levels: Sequence[float]) -> Dict[str, Any]:
    levels = [float(x) for x in reversal_levels]
    total_reversals = len(levels)

    if total_reversals >= TEST_REVERSAL_TARGET:
        used_levels = levels[-TEST_THRESHOLD_LAST_N:]
        threshold_mm = float(np.median(used_levels))
        return {
            "threshold_mm": threshold_mm,
            "threshold_kind": "formal",
            "threshold_label": "閾値",
            "threshold_detail": f"最後の{TEST_THRESHOLD_LAST_N} reversal の中央値",
            "threshold_used_levels": used_levels,
        }

    if total_reversals >= 6:
        used_levels = levels[4:]
        threshold_mm = float(np.median(used_levels))
        return {
            "threshold_mm": threshold_mm,
            "threshold_kind": "reference",
            "threshold_label": "参考値",
            "threshold_detail": "first 4 reversals を除外後、残り reversal の中央値",
            "threshold_used_levels": used_levels,
        }

    return {
        "threshold_mm": None,
        "threshold_kind": None,
        "threshold_label": "参考値",
        "threshold_detail": "6 reversal 未満のため算出なし",
        "threshold_used_levels": [],
    }


def build_test_progress_chart(df_logs: pd.DataFrame, phase_run: int) -> Optional[Dict[str, Any]]:
    test_df = df_logs[(df_logs["phase"] == "test") & (df_logs["phase_run"] == int(phase_run))].copy()
    if test_df.empty:
        return None

    test_df["phase_trial"] = pd.to_numeric(test_df["phase_trial"], errors="coerce")
    test_df["size_mm"] = pd.to_numeric(test_df["size_mm"], errors="coerce")
    test_df = test_df.dropna(subset=["phase_trial", "size_mm"])
    if test_df.empty:
        return None

    reversal_df = test_df[test_df["reversal"] == True].copy()  # noqa: E712
    reversal_values: List[Dict[str, Any]] = []
    if not reversal_df.empty:
        reversal_df["reversal_level_mm"] = pd.to_numeric(reversal_df["reversal_level_mm"], errors="coerce")
        reversal_df = reversal_df.dropna(subset=["reversal_level_mm"])
        reversal_df["reversal_marker_group"] = np.where(
            pd.to_numeric(reversal_df["n_reversals"], errors="coerce") <= 4,
            "first 4 reversal",
            "after first 4",
        )
        reversal_values = reversal_df[
            ["phase_trial", "reversal_level_mm", "n_reversals", "reversal_marker_group"]
        ].rename(columns={"reversal_level_mm": "size_mm"}).to_dict(orient="records")

    line_values = test_df.replace({np.nan: None}).to_dict(orient="records")
    point_values = pd.DataFrame(reversal_values).replace({np.nan: None}).to_dict(orient="records")

    return {
        "height": 320,
        "layer": [
            {
                "data": {"values": line_values},
                "mark": {"type": "line", "color": "#475569", "strokeWidth": 2.5},
                "encoding": {
                    "x": {"field": "phase_trial", "type": "quantitative", "title": "施行回数"},
                    "y": {"field": "size_mm", "type": "quantitative", "title": "dome サイズ (mm)"},
                    "tooltip": [
                        {"field": "phase_trial", "type": "quantitative", "title": "trial"},
                        {"field": "size_mm", "type": "quantitative", "title": "size", "format": ".2f"},
                        {"field": "orientation_presented", "type": "nominal", "title": "提示"},
                        {"field": "answer", "type": "nominal", "title": "回答"},
                        {"field": "correct", "type": "nominal", "title": "正誤"},
                    ],
                },
            },
            {
                "data": {"values": point_values},
                "mark": {"type": "point", "shape": "square", "filled": True, "size": 130},
                "encoding": {
                    "x": {"field": "phase_trial", "type": "quantitative", "title": "施行回数"},
                    "y": {"field": "size_mm", "type": "quantitative", "title": "dome サイズ (mm)"},
                    "color": {
                        "field": "reversal_marker_group",
                        "type": "nominal",
                        "title": "reversal",
                        "scale": {
                            "domain": ["first 4 reversal", "after first 4"],
                            "range": ["#2563eb", "#dc2626"],
                        },
                    },
                    "tooltip": [
                        {"field": "phase_trial", "type": "quantitative", "title": "trial"},
                        {"field": "size_mm", "type": "quantitative", "title": "reversal level", "format": ".2f"},
                        {"field": "n_reversals", "type": "quantitative", "title": "reversal count"},
                    ],
                },
            },
        ],
    }


def build_test_summary_text(summary: Dict[str, Any]) -> str:
    threshold_mm = summary.get("threshold_mm")
    nearest_dome_mm = summary.get("nearest_dome_mm")
    threshold_label = str(summary.get("threshold_label", "閾値"))
    threshold_detail = str(summary.get("threshold_detail", ""))
    threshold_used_levels = summary.get("threshold_used_levels") or []
    reversal_levels = summary.get("reversal_levels") or []
    series_seed = summary.get("series_seed")

    lines = [
        "Two-Point Orientation Discrimination - Main Test Summary",
        "",
        f"部位: {summary.get('selected_site', '')}",
        f"判定: {summary.get('result_label', '')}",
        f"理由: {summary.get('reason_text', '')}",
        f"trial数: {summary.get('trials', '')}",
        f"系列: {summary.get('series_name', '')}",
        f"seed: {'—' if series_seed is None else series_seed}",
        f"reversals: {summary.get('reversals', '')}",
    ]

    if threshold_mm is None:
        lines.append(f"{threshold_label}: —")
    else:
        lines.append(f"{threshold_label}: {format_mm(float(threshold_mm))} mm")
        lines.append(f"{threshold_label}の算出法: {threshold_detail}")
        lines.append(f"近い dome: {format_mm(float(nearest_dome_mm))} mm")

    if reversal_levels:
        lines.append("reversal levels: " + ", ".join(format_mm(float(x)) for x in reversal_levels))
    if threshold_used_levels:
        lines.append(
            f"{threshold_label}に使用した reversal levels: "
            + ", ".join(format_mm(float(x)) for x in threshold_used_levels)
        )

    lines.extend(
        [
            "",
            "1 = 縦",
            "2 = 横",
            "",
            "[codes]",
            format_schedule_codes(summary.get("schedule") or []),
            "",
            "[labels]",
            format_schedule_labels(summary.get("schedule") or []),
            "",
        ]
    )
    return "\n".join(lines)


@dataclass
class DiscreteTwoDownOneUpStaircase:
    levels: List[float]
    start_index: int

    index: int = field(init=False)
    n_correct_streak: int = field(default=0, init=False)
    last_direction: Optional[str] = field(default=None, init=False)
    reversals: List[Dict[str, Any]] = field(default_factory=list, init=False)
    update_n: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("levels must not be empty")
        if not (0 <= int(self.start_index) < len(self.levels)):
            raise ValueError("start_index out of range")
        self.index = int(self.start_index)

    def current_level(self) -> float:
        return float(self.levels[self.index])

    def update(self, correct: bool, phase_trial: int) -> Dict[str, Any]:
        self.update_n += 1
        prev_index = int(self.index)
        prev_level = float(self.levels[prev_index])
        direction: Optional[str] = None

        if bool(correct):
            self.n_correct_streak += 1
            if self.n_correct_streak >= 2:
                direction = "down"
                self.index = max(0, self.index - 1)
                self.n_correct_streak = 0
        else:
            self.n_correct_streak = 0
            direction = "up"
            self.index = min(len(self.levels) - 1, self.index + 1)

        reversal = False
        reversal_level: Optional[float] = None
        if direction is not None and self.last_direction is not None and direction != self.last_direction:
            reversal = True
            reversal_level = prev_level
            self.reversals.append(
                {
                    "phase_trial": int(phase_trial),
                    "update_n": int(self.update_n),
                    "index": int(prev_index),
                    "level": float(prev_level),
                }
            )

        if direction is not None:
            self.last_direction = direction

        return {
            "prev_index": int(prev_index),
            "prev_level": float(prev_level),
            "new_index": int(self.index),
            "new_level": float(self.levels[self.index]),
            "direction": direction,
            "reversal": bool(reversal),
            "reversal_level": reversal_level,
            "correct_streak_after_update": int(self.n_correct_streak),
        }

    def reversal_levels(self) -> List[float]:
        return [float(x["level"]) for x in self.reversals]

    def threshold_median(self, last_n: int) -> Optional[float]:
        levels = self.reversal_levels()
        if len(levels) < int(last_n):
            return None
        return float(np.median(levels[-int(last_n) :]))


@dataclass
class EasyPhaseState:
    phase: str
    phase_run: int
    seed: int
    schedule: List[int]
    levels: List[float] = field(default_factory=lambda: list(PRACTICE_LEVELS_MM))
    level_index: int = 0
    trial_n: int = 0
    correct_streak: int = 0
    errors_at_current_level: int = 0
    total_errors: int = 0

    def current_level(self) -> float:
        return float(self.levels[self.level_index])

    def current_orientation_code(self) -> int:
        return int(self.schedule[self.trial_n])

    def current_orientation_label(self) -> str:
        return orientation_label(self.current_orientation_code())


@dataclass
class MainPhaseState:
    phase_run: int
    series_name: str
    series_seed: Optional[int]
    schedule: List[int]
    staircase: DiscreteTwoDownOneUpStaircase
    trial_n: int = 0
    floor_correct_streak: int = 0
    ceil_incorrect_streak: int = 0

    def current_level(self) -> float:
        return self.staircase.current_level()

    def current_orientation_code(self) -> int:
        return int(self.schedule[self.trial_n])

    def current_orientation_label(self) -> str:
        return orientation_label(self.current_orientation_code())


def init_state() -> None:
    defaults: Dict[str, Any] = {
        "mode": "idle",
        "last_feedback": None,
        "ui_error": None,
        "undo_stack": [],
        "logs": [],
        "phase_runs": {"practice": 0, "test": 0, "post": 0},
        "phase_summaries": {"practice": None, "test": None, "post": None},
        "phase_summary_history": {"practice": [], "test": [], "post": []},
        "practice_state": None,
        "test_state": None,
        "post_state": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_all() -> None:
    for key in list(st.session_state.keys()):
        st.session_state.pop(key, None)
    init_state()

def ui_snapshot() -> Dict[str, Any]:
    return {
        "selected_hand": st.session_state.get("selected_hand", "右手"),
        "selected_finger": st.session_state.get("selected_finger", "示指"),
        "main_series_name": st.session_state.get("main_series_name", "系列1"),
        "main_random_seed_input": st.session_state.get("main_random_seed_input", ""),
    }


def selected_site_label(hand: Optional[str] = None, finger: Optional[str] = None) -> str:
    hand_text = str(hand if hand is not None else st.session_state.get("selected_hand", "右手"))
    finger_text = str(finger if finger is not None else st.session_state.get("selected_finger", "示指"))
    return f"{hand_text} / {finger_text}"


def common_log_fields() -> Dict[str, Any]:
    snap = ui_snapshot()
    hand = str(snap.get("selected_hand", "右手"))
    finger = str(snap.get("selected_finger", "示指"))
    return {
        "selected_hand": hand,
        "selected_finger": finger,
        "selected_site": f"{hand}-{finger}",
        "all_dome_levels_mm": levels_text(DOME_LEVELS_MM),
    }


def next_run_number(phase: str) -> int:
    runs = dict(st.session_state.get("phase_runs") or {})
    runs[str(phase)] = int(runs.get(str(phase), 0)) + 1
    st.session_state["phase_runs"] = runs
    return int(runs[str(phase)])


def begin_phase(phase: str) -> int:
    run_no = next_run_number(phase)
    st.session_state["mode"] = phase
    st.session_state["last_feedback"] = None
    st.session_state["ui_error"] = None
    st.session_state["undo_stack"] = []
    for phase_name in PHASE_LABELS:
        st.session_state[f"{phase_name}_state"] = None
    return run_no


def record_phase_summary(phase: str, summary: Dict[str, Any]) -> None:
    phase_summaries = dict(st.session_state.get("phase_summaries") or {})
    phase_summaries[phase] = summary
    st.session_state["phase_summaries"] = phase_summaries

    history = dict(st.session_state.get("phase_summary_history") or {})
    phase_history = list(history.get(phase) or [])
    phase_history.append(summary)
    history[phase] = phase_history
    st.session_state["phase_summary_history"] = history


def get_test_schedule(series_name: str, raw_seed: str) -> tuple[List[int], Optional[int]]:
    if series_name == "系列1":
        return list(TEST_SERIES_1), None
    if series_name == "系列2":
        return list(TEST_SERIES_2), None
    if series_name == "ランダム":
        seed = choose_seed(raw_seed, salt=777)
        return generate_random_series(TEST_MAX_TRIALS, seed), int(seed)
    raise ValueError(f"未知の系列です: {series_name}")


def start_easy_phase(phase: str) -> None:
    run_no = begin_phase(phase)
    salt = 1000 if phase == "practice" else 2000
    seed = choose_seed("", salt=run_no + salt)
    schedule = generate_random_series(PRACTICE_SCHEDULE_LEN, seed)
    st.session_state[f"{phase}_state"] = EasyPhaseState(
        phase=phase,
        phase_run=run_no,
        seed=seed,
        schedule=schedule,
    )


def start_test() -> None:
    snap = ui_snapshot()
    series_name = str(snap.get("main_series_name", "系列1"))
    raw_seed = str(snap.get("main_random_seed_input", ""))
    schedule, actual_seed = get_test_schedule(series_name, raw_seed)
    run_no = begin_phase("test")
    start_index = DOME_LEVELS_MM.index(TEST_START_MM)
    staircase = DiscreteTwoDownOneUpStaircase(levels=list(DOME_LEVELS_MM), start_index=start_index)
    st.session_state["test_state"] = MainPhaseState(
        phase_run=run_no,
        series_name=series_name,
        series_seed=actual_seed,
        schedule=schedule,
        staircase=staircase,
    )


def finalize_easy_phase(phase: str, reason: str) -> None:
    state: Optional[EasyPhaseState] = st.session_state.get(f"{phase}_state")
    if state is None:
        st.session_state["mode"] = "idle"
        return

    summary = {
        **common_log_fields(),
        "phase": phase,
        "phase_label": PHASE_LABELS[phase],
        "phase_run": int(state.phase_run),
        "result_label": "PASS" if reason == "pass" else ("FAIL" if reason == "fail" else "中止"),
        "reason_code": reason,
        "reason_text": PHASE_REASON_TEXT[phase][reason],
        "trials": int(state.trial_n),
        "seed": int(state.seed),
        "final_level_mm": float(state.current_level()),
        "correct_streak_end": int(state.correct_streak),
        "total_errors": int(state.total_errors),
        "completed_at_unix": float(time.time()),
    }

    record_phase_summary(phase, summary)
    st.session_state[f"{phase}_state"] = None
    st.session_state["mode"] = "idle"


def finalize_test_phase(reason: str) -> None:
    state: Optional[MainPhaseState] = st.session_state.get("test_state")
    if state is None:
        st.session_state["mode"] = "idle"
        return

    staircase = state.staircase
    threshold_summary = compute_test_threshold_summary(staircase.reversal_levels())
    threshold_mm = threshold_summary["threshold_mm"]
    nearest_dome_mm = nearest_level(staircase.levels, threshold_mm)

    if reason == "floor_correct_pass":
        result_label = "PASS"
    elif reason == "ceil_incorrect_fail":
        result_label = "FAIL"
    elif reason == "reversal_target":
        result_label = "収束完了"
    elif reason == "max_trials_nonconvergent":
        result_label = "収束不良"
    else:
        result_label = "中止"

    summary = {
        **common_log_fields(),
        "phase": "test",
        "phase_label": PHASE_LABELS["test"],
        "phase_run": int(state.phase_run),
        "result_label": result_label,
        "reason_code": reason,
        "reason_text": PHASE_REASON_TEXT["test"][reason],
        "trials": int(state.trial_n),
        "series_name": state.series_name,
        "series_seed": state.series_seed,
        "schedule": list(state.schedule),
        "reversals": int(len(staircase.reversals)),
        "reversal_levels": staircase.reversal_levels(),
        "threshold_mm": threshold_mm,
        "threshold_kind": threshold_summary["threshold_kind"],
        "threshold_label": threshold_summary["threshold_label"],
        "threshold_detail": threshold_summary["threshold_detail"],
        "threshold_used_levels": list(threshold_summary["threshold_used_levels"]),
        "nearest_dome_mm": nearest_dome_mm,
        "final_level_mm": float(staircase.current_level()),
        "floor_correct_streak_end": int(state.floor_correct_streak),
        "ceil_incorrect_streak_end": int(state.ceil_incorrect_streak),
        "completed_at_unix": float(time.time()),
    }

    record_phase_summary("test", summary)
    st.session_state["test_state"] = None
    st.session_state["mode"] = "idle"


def format_start_error(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return f"開始できませんでした: {exc}"
    return "開始できませんでした。設定を見直して再試行してください。"


def stop_active_phase() -> None:
    mode = st.session_state.get("mode")
    if mode == "practice":
        st.session_state["last_feedback"] = "練習を手動終了しました。"
        finalize_easy_phase("practice", "manual_stop")
    elif mode == "post":
        st.session_state["last_feedback"] = "事後を手動終了しました。"
        finalize_easy_phase("post", "manual_stop")
    elif mode == "test":
        st.session_state["last_feedback"] = "本番を手動終了しました。"
        finalize_test_phase("manual_stop")


UNDO_STATE_KEYS = [
    "mode",
    "last_feedback",
    "ui_error",
    "logs",
    "phase_runs",
    "phase_summaries",
    "phase_summary_history",
    "practice_state",
    "test_state",
    "post_state",
]


def push_undo_snapshot() -> None:
    snapshot = {key: copy.deepcopy(st.session_state.get(key)) for key in UNDO_STATE_KEYS}
    undo_stack = list(st.session_state.get("undo_stack") or [])
    undo_stack.append(snapshot)
    st.session_state["undo_stack"] = undo_stack


def can_undo_last_answer() -> bool:
    return bool(st.session_state.get("undo_stack"))


def undo_last_answer() -> None:
    undo_stack = list(st.session_state.get("undo_stack") or [])
    if not undo_stack:
        return

    snapshot = undo_stack.pop()
    for key, value in snapshot.items():
        st.session_state[key] = value
    st.session_state["undo_stack"] = undo_stack
    st.session_state["last_feedback"] = "直前の回答を取り消しました。"


def current_trial_info() -> Optional[Dict[str, Any]]:
    mode = st.session_state.get("mode")

    if mode in ("practice", "post"):
        state: Optional[EasyPhaseState] = st.session_state.get(f"{mode}_state")
        if state is None or state.trial_n >= len(state.schedule):
            return None
        return {
            "phase": mode,
            "phase_label": PHASE_LABELS[mode],
            "phase_run": int(state.phase_run),
            "trial_index": int(state.trial_n + 1),
            "size_mm": float(state.current_level()),
            "orientation_code": int(state.current_orientation_code()),
            "orientation_label": state.current_orientation_label(),
        }

    if mode == "test":
        state: Optional[MainPhaseState] = st.session_state.get("test_state")
        if state is None or state.trial_n >= len(state.schedule):
            return None
        return {
            "phase": "test",
            "phase_label": PHASE_LABELS["test"],
            "phase_run": int(state.phase_run),
            "trial_index": int(state.trial_n + 1),
            "size_mm": float(state.current_level()),
            "orientation_code": int(state.current_orientation_code()),
            "orientation_label": state.current_orientation_label(),
        }

    return None

def append_log_row(row: Dict[str, Any]) -> None:
    logs = list(st.session_state.get("logs") or [])
    logs.append(row)
    st.session_state["logs"] = logs


def handle_easy_answer(answer_label: str) -> None:
    mode = st.session_state.get("mode")
    if mode not in ("practice", "post"):
        return

    state: Optional[EasyPhaseState] = st.session_state.get(f"{mode}_state")
    if state is None:
        return

    push_undo_snapshot()

    phase_label = PHASE_LABELS[mode]
    trial_index = int(state.trial_n + 1)
    presented_code = int(state.current_orientation_code())
    presented_label = state.current_orientation_label()
    size_before = float(state.current_level())
    level_index_before = int(state.level_index)

    correct = str(answer_label) == str(presented_label)
    stepped_up = False
    end_reason: Optional[str] = None

    if correct:
        state.correct_streak += 1
    else:
        state.correct_streak = 0
        state.total_errors += 1
        state.errors_at_current_level += 1
        if state.errors_at_current_level >= PRACTICE_ERRORS_TO_STEP:
            if state.level_index >= len(state.levels) - 1:
                end_reason = "fail"
            else:
                state.level_index += 1
                state.errors_at_current_level = 0
                stepped_up = True

    if end_reason is None and state.correct_streak >= PRACTICE_PASS_STREAK:
        end_reason = "pass"

    next_size = float(state.current_level())

    append_log_row(
        {
            **common_log_fields(),
            "timestamp_unix": float(time.time()),
            "phase": mode,
            "phase_run": int(state.phase_run),
            "phase_trial": int(trial_index),
            "schedule_name": f"{phase_label}ランダム",
            "schedule_seed": int(state.seed),
            "orientation_presented_code": int(presented_code),
            "orientation_presented": presented_label,
            "size_mm": float(size_before),
            "answer": str(answer_label),
            "answer_code": int(orientation_code(answer_label)),
            "correct": bool(correct),
            "level_index_before": int(level_index_before),
            "level_index_after": int(state.level_index),
            "direction": "up" if stepped_up else "stay",
            "reversal": None,
            "reversal_level_mm": None,
            "n_reversals": None,
            "two_down_correct_streak_after": None,
            "easy_correct_streak_after": int(state.correct_streak),
            "errors_at_level_after": int(state.errors_at_current_level),
            "total_errors_easy": int(state.total_errors),
            "floor_correct_streak": None,
            "ceil_incorrect_streak": None,
            "next_size_mm": float(next_size),
            "threshold_live_mm": None,
            "threshold_live_nearest_dome_mm": None,
            "result_after_trial": PHASE_REASON_TEXT[mode][end_reason] if end_reason else "",
        }
    )

    state.trial_n += 1

    if end_reason == "pass":
        st.session_state["last_feedback"] = f"{phase_label}: ⭕ 正解 / 5問連続正答でPASS"
        finalize_easy_phase(mode, end_reason)
        return

    if end_reason == "fail":
        st.session_state["last_feedback"] = f"{phase_label}: ❌ 不正解 / 12 mm で2問誤答のためFAIL"
        finalize_easy_phase(mode, end_reason)
        return

    if correct:
        st.session_state["last_feedback"] = (
            f"{phase_label}: ⭕ 正解 / 提示: {format_mm(size_before)} mm・{presented_label}"
            f" / 次も {format_mm(next_size)} mm"
        )
    elif stepped_up:
        st.session_state["last_feedback"] = (
            f"{phase_label}: ❌ 不正解 / 提示: {format_mm(size_before)} mm・{presented_label}"
            f" / 次は {format_mm(next_size)} mm に上げます"
        )
    else:
        st.session_state["last_feedback"] = (
            f"{phase_label}: ❌ 不正解 / 提示: {format_mm(size_before)} mm・{presented_label}"
            f" / 次も {format_mm(next_size)} mm"
        )


def handle_test_answer(answer_label: str) -> None:
    if st.session_state.get("mode") != "test":
        return

    state: Optional[MainPhaseState] = st.session_state.get("test_state")
    if state is None:
        return

    push_undo_snapshot()

    staircase = state.staircase
    trial_index = int(state.trial_n + 1)
    presented_code = int(state.current_orientation_code())
    presented_label = state.current_orientation_label()
    size_before = float(staircase.current_level())
    at_floor = staircase.index == 0
    at_ceil = staircase.index == len(staircase.levels) - 1

    correct = str(answer_label) == str(presented_label)
    upd = staircase.update(correct=correct, phase_trial=trial_index)

    if at_floor:
        state.floor_correct_streak = state.floor_correct_streak + 1 if correct else 0
    else:
        state.floor_correct_streak = 0

    if at_ceil:
        state.ceil_incorrect_streak = state.ceil_incorrect_streak + 1 if not correct else 0
    else:
        state.ceil_incorrect_streak = 0

    threshold_live = staircase.threshold_median(TEST_THRESHOLD_LAST_N)
    threshold_nearest = nearest_level(staircase.levels, threshold_live)

    end_reason: Optional[str] = None
    if state.floor_correct_streak >= TEST_FLOOR_PASS_STREAK:
        end_reason = "floor_correct_pass"
    elif state.ceil_incorrect_streak >= TEST_CEIL_FAIL_STREAK:
        end_reason = "ceil_incorrect_fail"
    elif len(staircase.reversals) >= TEST_REVERSAL_TARGET:
        end_reason = "reversal_target"
    elif trial_index >= TEST_MAX_TRIALS:
        end_reason = "max_trials_nonconvergent"

    append_log_row(
        {
            **common_log_fields(),
            "timestamp_unix": float(time.time()),
            "phase": "test",
            "phase_run": int(state.phase_run),
            "phase_trial": int(trial_index),
            "schedule_name": state.series_name,
            "schedule_seed": state.series_seed,
            "orientation_presented_code": int(presented_code),
            "orientation_presented": presented_label,
            "size_mm": float(size_before),
            "answer": str(answer_label),
            "answer_code": int(orientation_code(answer_label)),
            "correct": bool(correct),
            "level_index_before": int(upd["prev_index"]),
            "level_index_after": int(upd["new_index"]),
            "direction": upd["direction"],
            "reversal": bool(upd["reversal"]),
            "reversal_level_mm": upd["reversal_level"],
            "n_reversals": int(len(staircase.reversals)),
            "two_down_correct_streak_after": int(upd["correct_streak_after_update"]),
            "easy_correct_streak_after": None,
            "errors_at_level_after": None,
            "total_errors_easy": None,
            "floor_correct_streak": int(state.floor_correct_streak),
            "ceil_incorrect_streak": int(state.ceil_incorrect_streak),
            "next_size_mm": float(upd["new_level"]),
            "threshold_live_mm": threshold_live,
            "threshold_live_nearest_dome_mm": threshold_nearest,
            "result_after_trial": PHASE_REASON_TEXT["test"][end_reason] if end_reason else "",
        }
    )

    state.trial_n += 1

    if end_reason == "floor_correct_pass":
        st.session_state["last_feedback"] = "本番: ⭕ 正解 / 0.35 mm で4問連続正答のためPASS"
        finalize_test_phase(end_reason)
        return

    if end_reason == "ceil_incorrect_fail":
        st.session_state["last_feedback"] = "本番: ❌ 不正解 / 12 mm で2問連続誤答のためFAIL"
        finalize_test_phase(end_reason)
        return

    if end_reason == "reversal_target":
        st.session_state["last_feedback"] = "本番: 10 reversals に到達したため終了しました。"
        finalize_test_phase(end_reason)
        return

    if end_reason == "max_trials_nonconvergent":
        st.session_state["last_feedback"] = "本番: 100 trial に到達したため収束不良と判定しました。"
        finalize_test_phase(end_reason)
        return

    st.session_state["last_feedback"] = (
        f"本番: {'⭕ 正解' if correct else '❌ 不正解'}"
        f" / 提示: {format_mm(size_before)} mm・{presented_label}"
        f" / 次: {format_mm(float(upd['new_level']))} mm"
    )

def inject_css() -> None:
    st.markdown(
        """
<style>
.big-display-card {
    border-radius: 22px;
    padding: 1.25rem 1.45rem 1.2rem 1.45rem;
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    border: 1px solid #dbeafe;
    min-height: 250px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    overflow: hidden;
}
.big-display-card.orientation {
    background: linear-gradient(135deg, #faf5ff 0%, #f5f3ff 100%);
    border-color: #ddd6fe;
}
.big-display-label {
    font-size: 1.2rem;
    font-weight: 700;
    color: #374151;
    margin-bottom: 0.95rem;
}
.big-display-value-dome {
    font-size: clamp(3.8rem, 7.2vw, 5.6rem);
    line-height: 0.95;
    font-weight: 800;
    color: #1d4ed8;
    white-space: nowrap;
    letter-spacing: -0.04em;
    word-break: keep-all;
    overflow-wrap: normal;
}
.big-display-value-orientation {
    font-size: clamp(5rem, 10vw, 6.8rem);
    line-height: 0.95;
    font-weight: 800;
    color: #7c3aed;
    white-space: nowrap;
}
.big-display-value-dome .unit {
    font-size: 0.82em;
}
@media (max-width: 900px) {
    .big-display-card {
        min-height: 220px;
        padding: 1.1rem 1.2rem 1.05rem 1.2rem;
    }
    .big-display-value-dome {
        font-size: clamp(3.2rem, 9vw, 4.6rem);
    }
    .big-display-value-orientation {
        font-size: clamp(4.2rem, 11vw, 5.6rem);
    }
}
.summary-card {
    border-radius: 16px;
    padding: 1rem 1.05rem;
    border: 1px solid #e5e7eb;
    background: #fafafa;
    min-height: 165px;
}
.summary-title {
    font-size: 1rem;
    font-weight: 700;
    color: #374151;
    margin-bottom: 0.3rem;
}
.summary-result {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0 0 0.55rem 0;
}
.summary-meta {
    font-size: 0.95rem;
    line-height: 1.45;
    color: #4b5563;
}
.summary-pass .summary-result {
    color: #15803d;
}
.summary-fail .summary-result {
    color: #b91c1c;
}
.summary-complete .summary-result {
    color: #1d4ed8;
}
.summary-neutral .summary-result {
    color: #6b7280;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_big_display(title: str, value: str, kind: str) -> None:
    if kind == "orientation":
        extra_class = " orientation"
        value_class = "big-display-value-orientation"
        value_html = html.escape(value)
    else:
        extra_class = ""
        value_class = "big-display-value-dome"
        number_text = value.replace(" mm", "")
        value_html = (
            f'<span class="number">{html.escape(number_text)}</span>'
            f'<span class="unit">&nbsp;mm</span>'
        )

    st.markdown(
        f"""
<div class="big-display-card{extra_class}">
    <div class="big-display-label">{html.escape(title)}</div>
    <div class="{value_class}">{value_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def summary_card_class(summary: Optional[Dict[str, Any]]) -> str:
    if not summary:
        return "summary-neutral"
    result = str(summary.get("result_label", ""))
    if result == "PASS":
        return "summary-pass"
    if result in {"FAIL", "収束不良"}:
        return "summary-fail"
    if result == "収束完了":
        return "summary-complete"
    return "summary-neutral"


def render_phase_summary_card(phase: str, summary: Optional[Dict[str, Any]]) -> None:
    if not summary:
        st.markdown(
            f"""
<div class="summary-card summary-neutral">
    <div class="summary-title">{PHASE_LABELS[phase]}</div>
    <div class="summary-result">未実施</div>
    <div class="summary-meta">まだ記録がありません。</div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    meta_lines: List[str] = [
        f"{summary.get('reason_text', '')}",
        f"trial: {summary.get('trials', '—')}",
    ]

    if phase == "test":
        meta_lines.append(f"series: {summary.get('series_name', '—')}")
        meta_lines.append(f"reversals: {summary.get('reversals', '—')}")
        thr = summary.get("threshold_mm")
        thr_label = str(summary.get("threshold_label", "threshold"))
        if thr is None:
            meta_lines.append(f"{thr_label}: —")
        else:
            meta_lines.append(f"{thr_label}: {format_mm(float(thr))} mm")
    else:
        meta_lines.append(f"final dome: {format_mm(float(summary.get('final_level_mm', 0.0)))} mm")

    st.markdown(
        f"""
<div class="summary-card {summary_card_class(summary)}">
    <div class="summary-title">{summary.get('phase_label', PHASE_LABELS[phase])}</div>
    <div class="summary-result">{summary.get('result_label', '—')}</div>
    <div class="summary-meta">{'<br>'.join(meta_lines)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


st.set_page_config(page_title=APP_TITLE, page_icon="🖐️", layout="centered")
inject_css()
init_state()

st.title(APP_TITLE)
st.caption(
    "JVP dome 用の検査者補助アプリ。刺激時間は1秒。練習・本番・事後を必要に応じて開始し、縦/横回答と 2-down 1-up staircase を記録します。"
)

mode = st.session_state.get("mode", "idle")
ui_locked = mode != "idle"

with st.sidebar:
    st.header("⚙️ 設定")

    st.subheader("手と指")
    st.selectbox(
        "手",
        options=HAND_OPTIONS,
        index=HAND_OPTIONS.index("右手"),
        key="selected_hand",
        disabled=ui_locked,
    )
    st.selectbox(
        "指",
        options=FINGER_OPTIONS,
        index=FINGER_OPTIONS.index("示指"),
        key="selected_finger",
        disabled=ui_locked,
    )
    st.caption(f"現在の設定: {selected_site_label()}")

    st.subheader("本番系列")
    series_name = st.selectbox(
        "系列",
        options=["系列1", "系列2", "ランダム"],
        index=0,
        key="main_series_name",
        disabled=ui_locked,
    )
    st.text_input(
        "ランダム系列 seed（空欄で自動）",
        key="main_random_seed_input",
        disabled=ui_locked or series_name != "ランダム",
    )

    st.divider()
    st.markdown("**固定 dome サイズ（mm）**")
    st.caption(levels_text(DOME_LEVELS_MM))
    st.caption("Blank dome は使いません。")

    st.markdown("**練習 / 事後**")
    st.caption("刺激時間は1秒。8 mm 開始。5問連続正答で PASS。8 mm で2問誤答→10 mm、10 mm で2問誤答→12 mm、12 mm で2問誤答→FAIL。")

    st.markdown("**本番**")
    st.caption("刺激時間は1秒。8 mm 開始。2連続正答で1段階小さく、1回誤答で1段階大きく。0.35 mm で4連続正答=PASS、12 mm で2連続誤答=FAIL、10 reversals で終了、100 trial で収束不良。")

    st.divider()
    if st.button("リセット（全消去）"):
        reset_all()
        st.rerun()

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("練習", disabled=mode != "idle", use_container_width=True):
        start_easy_phase("practice")
        st.rerun()
with c2:
    if st.button("本番", disabled=mode != "idle", use_container_width=True):
        try:
            start_test()
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.session_state["ui_error"] = format_start_error(exc)
with c3:
    if st.button("事後", disabled=mode != "idle", use_container_width=True):
        start_easy_phase("post")
        st.rerun()
with c4:
    if st.button("終了", disabled=mode == "idle", use_container_width=True):
        stop_active_phase()
        st.rerun()

if mode == "idle":
    st.info("練習・本番・事後のいずれかを開始してください。")
elif mode == "practice":
    st.success("練習モード")
elif mode == "test":
    st.success("本番モード")
elif mode == "post":
    st.success("事後モード")

if st.session_state.get("ui_error"):
    st.error(str(st.session_state["ui_error"]))

st.caption(f"選択部位: {selected_site_label()}")

phase_summaries = st.session_state.get("phase_summaries") or {}
phase_summary_history = st.session_state.get("phase_summary_history") or {}
sc1, sc2, sc3 = st.columns(3)
with sc1:
    render_phase_summary_card("practice", phase_summaries.get("practice"))
with sc2:
    render_phase_summary_card("test", phase_summaries.get("test"))
with sc3:
    render_phase_summary_card("post", phase_summaries.get("post"))

trial = current_trial_info()

if mode in ("practice", "post"):
    state: Optional[EasyPhaseState] = st.session_state.get(f"{mode}_state")
    if state is not None:
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("trial", f"{state.trial_n + 1}")
        with m2:
            st.metric("現在mm", f"{format_mm(state.current_level())} mm")
        with m3:
            st.metric("連続正答", f"{state.correct_streak} / {PRACTICE_PASS_STREAK}")
        with m4:
            st.metric("このレベルの誤答", f"{state.errors_at_current_level} / {PRACTICE_ERRORS_TO_STEP}")
        with m5:
            st.metric("累積誤答", f"{state.total_errors}")

        st.caption(
            "練習/事後: 刺激時間は1秒。8 mm 開始、5問連続正答で PASS。"
            " 8 mm で2問誤答→10 mm、10 mm で2問誤答→12 mm、12 mm で2問誤答→FAIL。"
        )

if mode == "test":
    state = st.session_state.get("test_state")
    if state is not None:
        staircase = state.staircase
        thr_live = staircase.threshold_median(TEST_THRESHOLD_LAST_N)
        thr_nearest = nearest_level(staircase.levels, thr_live)

        m1, m2, m3, m4, m5 = st.columns([1.35, 1.15, 1.0, 1.0, 1.0])
        with m1:
            st.metric("trial", f"{state.trial_n + 1}/{TEST_MAX_TRIALS}")
        with m2:
            st.metric("現在mm", f"{format_mm(staircase.current_level())} mm")
        with m3:
            st.metric("reversals", f"{len(staircase.reversals)} / {TEST_REVERSAL_TARGET}")
        with m4:
            st.metric("暫定閾値", "—" if thr_live is None else f"{format_mm(thr_live)} mm")
        with m5:
            st.metric("近いdome", "—" if thr_nearest is None else f"{format_mm(thr_nearest)} mm")
        st.caption(f"系列: {state.series_name}")

        st.caption(
            "本番: 刺激時間は1秒。2連続正答で1段階小さく、1回誤答で1段階大きく。"
            f" 0.35 mm 連続正答 {state.floor_correct_streak}/{TEST_FLOOR_PASS_STREAK}、"
            f" 12 mm 連続誤答 {state.ceil_incorrect_streak}/{TEST_CEIL_FAIL_STREAK}。"
        )

if trial is not None:
    if mode in ("practice", "post"):
        state = st.session_state.get(f"{mode}_state")
        streak = state.correct_streak if state is not None else 0
        st.subheader(f"{trial['phase_label']}: Trial {trial['trial_index']}（連続正答 {streak}/{PRACTICE_PASS_STREAK}）")
    elif mode == "test":
        st.subheader(f"本番: Trial {trial['trial_index']} / {TEST_MAX_TRIALS}")

    box1, box2 = st.columns([1.18, 0.82])
    with box1:
        render_big_display("次に使う dome", f"{format_mm(float(trial['size_mm']))} mm", kind="dome")
    with box2:
        render_big_display("次の向き", str(trial["orientation_label"]), kind="orientation")

    st.markdown("### 患者の回答")
    a1, a2 = st.columns(2)
    with a1:
        if st.button("縦", use_container_width=True):
            if mode in ("practice", "post"):
                handle_easy_answer("縦")
            elif mode == "test":
                handle_test_answer("縦")
            st.rerun()
    with a2:
        if st.button("横", use_container_width=True):
            if mode in ("practice", "post"):
                handle_easy_answer("横")
            elif mode == "test":
                handle_test_answer("横")
            st.rerun()
    if st.button("ひとつ前に戻る", use_container_width=True, disabled=not can_undo_last_answer()):
        undo_last_answer()
        st.rerun()

elif can_undo_last_answer():
    st.caption("終了直後でも、誤入力なら直前の回答を取り消して再開できます。")
    if st.button("ひとつ前に戻る", use_container_width=True):
        undo_last_answer()
        st.rerun()

if st.session_state.get("last_feedback"):
    st.write(st.session_state["last_feedback"])

logs = st.session_state.get("logs") or []

if any(value is not None for value in phase_summaries.values()):
    st.divider()
    st.subheader("詳細結果")
    st.caption("各フェーズの詳細は最新 run を表示しています。過去 run は下の履歴で確認できます。")

    practice_summary = phase_summaries.get("practice")
    if practice_summary is not None:
        with st.expander("練習の詳細", expanded=False):
            st.write(f"部位: **{practice_summary.get('selected_site', selected_site_label(practice_summary.get('selected_hand'), practice_summary.get('selected_finger')))}**")
            st.write(f"判定: **{practice_summary['result_label']}**")
            st.write(f"理由: **{practice_summary['reason_text']}**")
            st.write(f"trial数: **{practice_summary['trials']}**")
            st.write(f"最終 dome: **{format_mm(practice_summary['final_level_mm'])} mm**")
            st.write(f"乱数 seed: **{practice_summary['seed']}**")

    test_summary = phase_summaries.get("test")
    if test_summary is not None:
        with st.expander("本番の詳細", expanded=True):
            st.write(f"部位: **{test_summary.get('selected_site', selected_site_label(test_summary.get('selected_hand'), test_summary.get('selected_finger')))}**")
            st.write(f"判定: **{test_summary['result_label']}**")
            st.write(f"理由: **{test_summary['reason_text']}**")
            st.write(f"trial数: **{test_summary['trials']}**")
            st.write(f"系列: **{test_summary['series_name']}**")
            if test_summary.get("series_seed") is not None:
                st.write(f"seed: **{test_summary['series_seed']}**")
            st.write(f"reversals: **{test_summary['reversals']}**")
            if test_summary.get("threshold_mm") is None:
                st.write(f"{test_summary.get('threshold_label', '参考値')}: **—**（まだ6 reversal 未満）")
            else:
                st.write(
                    f"{test_summary.get('threshold_label', '閾値')}（{test_summary.get('threshold_detail', '')}）: "
                    f"**{format_mm(test_summary['threshold_mm'])} mm**"
                )
                st.write(f"近い dome: **{format_mm(test_summary['nearest_dome_mm'])} mm**")
            reversal_levels = test_summary.get("reversal_levels") or []
            if reversal_levels:
                st.caption("reversal levels: " + ", ".join(format_mm(x) for x in reversal_levels))
            threshold_used_levels = test_summary.get("threshold_used_levels") or []
            if threshold_used_levels:
                st.caption(
                    f"{test_summary.get('threshold_label', '閾値')}に使用した reversal levels: "
                    + ", ".join(format_mm(x) for x in threshold_used_levels)
                )

            if logs:
                df_logs = pd.DataFrame(logs)
                chart = build_test_progress_chart(df_logs, int(test_summary["phase_run"]))
                if chart is not None:
                    st.vega_lite_chart(chart, use_container_width=True)
                    st.caption("青四角: first 4 reversals / 赤四角: それ以降の reversals")

            schedule_text = build_test_summary_text(test_summary)
            st.download_button(
                "本番結果と系列をテキストでダウンロード",
                data=schedule_text.encode("utf-8"),
                file_name="two_point_orientation_discrimination_test_summary.txt",
                mime="text/plain",
            )

    post_summary = phase_summaries.get("post")
    if post_summary is not None:
        with st.expander("事後の詳細", expanded=False):
            st.write(f"部位: **{post_summary.get('selected_site', selected_site_label(post_summary.get('selected_hand'), post_summary.get('selected_finger')))}**")
            st.write(f"判定: **{post_summary['result_label']}**")
            st.write(f"理由: **{post_summary['reason_text']}**")
            st.write(f"trial数: **{post_summary['trials']}**")
            st.write(f"最終 dome: **{format_mm(post_summary['final_level_mm'])} mm**")
            st.write(f"乱数 seed: **{post_summary['seed']}**")

if any(phase_summary_history.get(phase) for phase in PHASE_LABELS):
    st.divider()
    st.subheader("実施履歴")
    history_rows: List[Dict[str, Any]] = []
    for phase in PHASE_LABELS:
        for summary in phase_summary_history.get(phase) or []:
            row = {
                "phase": summary.get("phase_label", PHASE_LABELS[phase]),
                "run": summary.get("phase_run"),
                "result": summary.get("result_label"),
                "reason": summary.get("reason_text"),
                "trials": summary.get("trials"),
                "site": summary.get("selected_site"),
            }
            if phase == "test":
                row["series"] = summary.get("series_name")
                row["threshold_label"] = summary.get("threshold_label")
                row["threshold_mm"] = summary.get("threshold_mm")
                row["reversals"] = summary.get("reversals")
            else:
                row["final_level_mm"] = summary.get("final_level_mm")
            history_rows.append(row)
    history_df = pd.DataFrame(history_rows)
    st.dataframe(history_df, use_container_width=True, height=220)

if logs:
    st.divider()
    st.subheader("ログ")
    df = pd.DataFrame(logs)
    st.dataframe(df, use_container_width=True, height=320)
    st.download_button(
        "ログCSVをダウンロード",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="two_point_orientation_discrimination_log.csv",
        mime="text/csv",
    )
