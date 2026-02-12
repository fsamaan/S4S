"""Analysis history and undo/redo functionality."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import copy
import pandas as pd
import streamlit as st


class AnalysisHistory:
    """Manages analysis history for undo/redo functionality."""

    MAX_HISTORY = 20  # Maximum number of states to keep

    def __init__(self):
        self._history: List[Dict[str, Any]] = []
        self._current_index: int = -1
        self._analysis_log: List[Dict[str, Any]] = []

    def save_state(self, state: Dict[str, Any], action: str, details: Optional[Dict] = None) -> None:
        """Save a state snapshot to history.

        Args:
            state: Dictionary of state values to save
            action: Description of the action that led to this state
            details: Additional details about the action
        """
        # Remove any future states if we're not at the end
        if self._current_index < len(self._history) - 1:
            self._history = self._history[:self._current_index + 1]

        # Create snapshot (deep copy to avoid mutations)
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "details": details or {},
            "state": self._serialize_state(state),
        }

        self._history.append(snapshot)
        self._current_index = len(self._history) - 1

        # Trim history if too long
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[-self.MAX_HISTORY:]
            self._current_index = len(self._history) - 1

        # Log the event
        self._analysis_log.append({
            "timestamp": snapshot["timestamp"],
            "event": action,
            "details": details or {},
        })

    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize state for storage, handling DataFrames specially."""
        serialized = {}
        for key, value in state.items():
            if isinstance(value, pd.DataFrame):
                serialized[key] = {
                    "_type": "DataFrame",
                    "data": value.to_dict(),
                    "attrs": getattr(value, "attrs", {}),
                }
            elif isinstance(value, dict):
                # Recursively serialize nested dicts
                serialized[key] = self._serialize_state(value)
            else:
                try:
                    # Try to deep copy, fall back to reference
                    serialized[key] = copy.deepcopy(value)
                except Exception:
                    serialized[key] = value
        return serialized

    def _deserialize_state(self, serialized: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize state, reconstructing DataFrames."""
        state = {}
        for key, value in serialized.items():
            if isinstance(value, dict):
                if value.get("_type") == "DataFrame":
                    df = pd.DataFrame.from_dict(value["data"])
                    df.attrs = value.get("attrs", {})
                    state[key] = df
                else:
                    state[key] = self._deserialize_state(value)
            else:
                state[key] = value
        return state

    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return self._current_index > 0

    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return self._current_index < len(self._history) - 1

    def undo(self) -> Optional[Dict[str, Any]]:
        """Undo to previous state.

        Returns:
            Previous state dictionary or None if can't undo
        """
        if not self.can_undo():
            return None

        self._current_index -= 1
        return self._deserialize_state(self._history[self._current_index]["state"])

    def redo(self) -> Optional[Dict[str, Any]]:
        """Redo to next state.

        Returns:
            Next state dictionary or None if can't redo
        """
        if not self.can_redo():
            return None

        self._current_index += 1
        return self._deserialize_state(self._history[self._current_index]["state"])

    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get current state."""
        if self._current_index < 0 or self._current_index >= len(self._history):
            return None
        return self._deserialize_state(self._history[self._current_index]["state"])

    def get_history_summary(self) -> List[Dict[str, str]]:
        """Get a summary of the history for display."""
        summary = []
        for i, item in enumerate(self._history):
            summary.append({
                "index": i,
                "timestamp": item["timestamp"],
                "action": item["action"],
                "current": i == self._current_index,
            })
        return summary

    def get_analysis_log(self) -> List[Dict[str, Any]]:
        """Get the full analysis log."""
        return self._analysis_log

    def clear_history(self) -> None:
        """Clear all history."""
        self._history = []
        self._current_index = -1


def get_history() -> AnalysisHistory:
    """Get or create the analysis history singleton in session state."""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = AnalysisHistory()
    return st.session_state.analysis_history


def save_analysis_state(action: str, details: Optional[Dict] = None) -> None:
    """Save current analysis state to history.

    Args:
        action: Description of the action
        details: Additional details
    """
    history = get_history()

    # Collect relevant state
    state_keys = [
        "df", "raw_df", "data_dictionary", "table1_pub", "table1_raw",
        "table2", "table2_or", "model_results", "model_compare",
        "selected_covariates", "table1_exposure", "exposure_reference",
        "time_event_cols", "selected_outcomes", "selected_outcome_types",
    ]

    state = {}
    for key in state_keys:
        if key in st.session_state:
            value = st.session_state[key]
            if value is not None:
                state[key] = value

    history.save_state(state, action, details)


def restore_state(state: Dict[str, Any]) -> None:
    """Restore session state from a saved state.

    Args:
        state: State dictionary to restore
    """
    for key, value in state.items():
        st.session_state[key] = value


def undo_action() -> bool:
    """Undo the last action.

    Returns:
        True if undo was successful
    """
    history = get_history()
    if not history.can_undo():
        return False

    state = history.undo()
    if state:
        restore_state(state)
        return True
    return False


def redo_action() -> bool:
    """Redo the previously undone action.

    Returns:
        True if redo was successful
    """
    history = get_history()
    if not history.can_redo():
        return False

    state = history.redo()
    if state:
        restore_state(state)
        return True
    return False


def log_event(event: str, details: Dict[str, Any]) -> None:
    """Log an analysis event (legacy compatibility).

    Args:
        event: Event name
        details: Event details
    """
    history = get_history()
    history._analysis_log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "details": details,
    })
