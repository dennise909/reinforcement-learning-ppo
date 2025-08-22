import numpy as np
import pandas as pd


SLOT_MIN = 15


def utilization(grid: np.ndarray) -> float:
    free = (grid == 0).sum()
    total = grid.size
    used = total - free
    return used / total if total > 0 else 0.0


def count_large_gaps(grid: np.ndarray, min_slots=3) -> int:
    def _gaps(row):
        gaps = 0
        idx = np.where(row == 1)[0]
        if len(idx) < 2:
            return 0
        for i in range(len(idx)-1):
            gap = idx[i+1] - idx[i] - 1
            if gap >= min_slots:
                gaps += 1
        return gaps
    return sum(_gaps(grid[d]) for d in range(grid.shape[0]))


def summarize(assignments, n_doctors, n_slots):
    # assignments: (patient_id, d, s, urgent, avail_start)
    grid = np.zeros((n_doctors, n_slots), dtype=int)
    for _, d, s, *_ in assignments:
        grid[d, s] = 1
    util = utilization(grid)
    gaps = count_large_gaps(grid)
    unassigned = None # set by caller si lo desea
    # latencia urgencias
    latencies = [s - astart for _, _, s, u, astart in assignments if u == 1]
    avg_urg_latency = float(np.mean(latencies)) if len(latencies) else 0.0
    return {
        "utilization": util,
        "large_gaps": gaps,
        "avg_urgent_latency": avg_urg_latency,
        "grid": grid,
    }


def assignments_to_df(assignments, day_start_min=9*60):
    rows = []
    for pid, d, s, urgent, _ in assignments:
        start_min = day_start_min + s * SLOT_MIN
        end_min = start_min + SLOT_MIN
        rows.append({
            "doctor": f"Doctor_{d}",
            "patient": f"Paciente_{pid}",
            "start_min": start_min,
            "end_min": end_min,
            "slot": s,
            "urgent": urgent,
            "tipo": "Urgencia" if urgent else "Normal",
        })
    return pd.DataFrame(rows)