"""Constraint-dominance NSGA-II selection.

Ranking rule (blueprint §6.4):

    feasible                      beats  infeasible
    among infeasible: smaller total violation beats larger
    among feasible:   Pareto non-dominated sort + crowding distance
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evoquant.search.objectives import CandidateEvaluation, TargetConfig


@dataclass(frozen=True)
class RankedCandidate:
    index: int  # position in the population list passed in
    feasible: bool
    total_violation: float
    front: int  # 0 = best; infeasible candidates get front = 10**6 tier
    crowding: float


def pareto_dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a dominates b (all objectives maximized)."""
    return bool(np.all(a >= b) and np.any(a > b))


def constraint_dominates(
    ea: CandidateEvaluation, eb: CandidateEvaluation, targets: TargetConfig
) -> bool:
    fa, fb = ea.is_feasible(targets), eb.is_feasible(targets)
    if fa and not fb:
        return True
    if not fa and fb:
        return False
    if not fa and not fb:
        return ea.total_violation(targets) < eb.total_violation(targets)
    return pareto_dominates(ea.objectives(), eb.objectives())


def fast_nondominated_sort(objs: np.ndarray) -> list[list[int]]:
    """Standard NSGA-II fronts over a (n, m) objective matrix (maximize)."""
    n = len(objs)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    dom_count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            if pareto_dominates(objs[i], objs[j]):
                dominated_by[i].append(j)
                dom_count[j] += 1
            elif pareto_dominates(objs[j], objs[i]):
                dominated_by[j].append(i)
                dom_count[i] += 1
    fronts: list[list[int]] = []
    current = [i for i in range(n) if dom_count[i] == 0]
    while current:
        fronts.append(current)
        nxt: list[int] = []
        for i in current:
            for j in dominated_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def crowding_distance(objs: np.ndarray, front: list[int]) -> dict[int, float]:
    dist = dict.fromkeys(front, 0.0)
    if len(front) <= 2:
        return dict.fromkeys(front, float("inf"))
    for m in range(objs.shape[1]):
        order = sorted(front, key=lambda i: objs[i, m])
        lo, hi = objs[order[0], m], objs[order[-1], m]
        dist[order[0]] = dist[order[-1]] = float("inf")
        if hi - lo <= 0:
            continue
        for k in range(1, len(order) - 1):
            dist[order[k]] += float(
                (objs[order[k + 1], m] - objs[order[k - 1], m]) / (hi - lo)
            )
    return dist


def rank_population(
    evals: list[CandidateEvaluation], targets: TargetConfig
) -> list[RankedCandidate]:
    """Rank candidates; result sorted best-first."""
    feas_idx = [i for i, e in enumerate(evals) if e.is_feasible(targets)]
    infeas_idx = [i for i in range(len(evals)) if i not in set(feas_idx)]

    ranked: list[RankedCandidate] = []
    if feas_idx:
        objs = np.vstack([evals[i].objectives() for i in feas_idx])
        fronts = fast_nondominated_sort(objs)
        for front_no, front in enumerate(fronts):
            crowd = crowding_distance(objs, front)
            for local_i in front:
                ranked.append(
                    RankedCandidate(
                        index=feas_idx[local_i],
                        feasible=True,
                        total_violation=0.0,
                        front=front_no,
                        crowding=crowd[local_i],
                    )
                )
    for i in infeas_idx:
        ranked.append(
            RankedCandidate(
                index=i,
                feasible=False,
                total_violation=evals[i].total_violation(targets),
                front=10**6,
                crowding=0.0,
            )
        )
    ranked.sort(
        key=lambda r: (
            not r.feasible,  # feasible first
            r.front,
            r.total_violation,
            -r.crowding,
            r.index,  # deterministic tiebreak
        )
    )
    return ranked


def select_survivors(
    evals: list[CandidateEvaluation], targets: TargetConfig, k: int
) -> list[int]:
    """Indices of the k best candidates under constraint dominance."""
    return [r.index for r in rank_population(evals, targets)[:k]]
