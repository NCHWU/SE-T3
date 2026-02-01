"""
Assignment 3 — Scenario-Based Testing of an RL Agent (Hill Climbing)

You MUST implement:
    - compute_objectives_from_time_series
    - compute_fitness
    - mutate_config
    - hill_climb

DO NOT change function signatures.
You MAY add helper functions.

Goal
----
Find a scenario (environment configuration) that triggers a collision.
If you cannot trigger a collision, minimize the minimum distance between the ego
vehicle and any other vehicle across the episode.

Black-box requirement
---------------------
Your evaluation must rely only on observable behavior during execution:
- crashed flag from the environment
- time-series data returned by run_episode (positions, lane_id, etc.)
No internal policy/model details beyond calling policy(obs, info).
"""

import copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from envs.highway_env_utils import run_episode, record_video_episode


# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

def compute_objectives_from_time_series(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute your objective values from the recorded time-series.

    The time_series is a list of frames. Each frame typically contains:
      - frame["crashed"]: bool
      - frame["ego"]: dict or None, e.g. {"pos":[x,y], "lane_id":..., "length":..., "width":...}
      - frame["others"]: list of dicts with positions, lane_id, etc.

    Minimum requirements (suggested):
      - crash_count: 1 if any collision happened, else 0
      - min_distance: minimum distance between ego and any other vehicle over time (float)

    Return a dictionary, e.g.:
        {
          "crash_count": 0 or 1,
          "min_distance": float
        }

    NOTE: If you want, you can add more objectives (lane-specific distances, time-to-crash, etc.)
    but keep the keys above at least.
    """
    # TODO (students)
    result = {}
    result = {
        "crash_count" : 0,
        "min_distance" : float('inf')
    }
    for t in time_series:
        if t['crashed']:
            result["crash_count"] = 1
            break
        # ego
        ego_x, ego_y = t["ego"]["pos"]
        e_speed, e_heading = t["ego"]["speed"], t["ego"]["heading"]
        e_length, e_width = t["ego"]["length"], t["ego"]["width"]
        o_lane_id = t["ego"]["lane_id"]

        # metrics to record
        min_distance = result["min_distance"]
        # other
        for other in t["others"]:
            o_x, o_y = other["pos"]
            o_length, o_width = other["length"], other["width"]
            o_lane_id = other["lane_id"]
            # compare ego with other vehicle (NAIVE: x for now)
            # abs distance
            diff = abs(o_x - ego_x)
            result["min_distance"] = min(result["min_distance"], diff)
    
    return result


def compute_fitness(objectives: Dict[str, Any]) -> float:
    """
    Convert objectives into ONE scalar fitness value to MINIMIZE.

    Requirement:
    - Any crashing scenario must be strictly better than any non-crashing scenario.

    Examples:
    - If crash_count==1: fitness = -1 (best)
    - Else: fitness = min_distance (smaller is better)

    You can design a more refined scalarization if desired.
    """
    fitness = 0
    if objectives["crash_count"] == 1:
        fitness = -1
    else:
        fitness = objectives["min_distance"]
    return fitness


# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================
"""
param_spec = {
    "vehicles_count":   {"type": "int",   "min": 5,   "max": 60},
    "lanes_count":      {"type": "int",   "min": 3,   "max": 10},
    "initial_spacing":  {"type": "float", "min": 0.5, "max": 5.0},
    "ego_spacing":      {"type": "float", "min": 1.0, "max": 4.0},
    "initial_lane_id":  {"type": "int",   "min": 0,   "max": 4},
}
"""
def mutate_config(
    cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Generate ONE neighbor configuration by mutating the current scenario.

    Inputs:
      - cfg: current scenario dict (e.g., vehicles_count, initial_spacing, ego_spacing, initial_lane_id)
      - param_spec: search space bounds, types (int/float), min/max
      - rng: random generator

    Requirements:
      - Do NOT modify cfg in-place (return a copy).
      - Keep mutated values within [min, max] from param_spec.
      - If you mutate lanes_count, keep initial_lane_id valid (0..lanes_count-1).

    Students can implement:
      - single-parameter mutation (recommended baseline)
      - multiple-parameter mutation
      - adaptive step sizes, etc.
    """
    # print("MUTATED_CONFIG: PARAM_SPEC", param_spec)
    mutated_config = cfg.copy()
    # print(mutated_config.keys())
    # naive single-parameter mutation based on randomization
    selections = ['vehicles_count', 'lanes_count', 'initial_spacing', 'ego_spacing', 'initial_lane_id']
    select = rng.choice(selections)
    m_type = param_spec[select]["type"]
    m_min, m_max = param_spec[select]["min"], param_spec[select]["max"]
    if m_type == "int":
        mutated_config[select] = rng.integers(m_min, m_max + 1)
    elif m_type == 'float':
        mutated_config[select] = rng.uniform(m_min, m_max)

    return mutated_config       


# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================

def hill_climb(
    env_id: str,
    base_cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    policy,
    defaults: Dict[str, Any],
    seed: int = 0,
    iterations: int = 100,
    neighbors_per_iter: int = 10,
) -> Dict[str, Any]:
    """
    Hill climbing loop.

    You should:
      1) Start from an initial scenario (base_cfg or random sample).
      2) Evaluate it by running:
            crashed, ts = run_episode(env_id, cfg, policy, defaults, seed_base)
        Then compute objectives + fitness.
      3) For each iteration:
            - Generate neighbors_per_iter neighbors using mutate_config
            - Evaluate each neighbor
            - Select the best neighbor
            - Accept it if it improves fitness (or implement another acceptance rule)
            - Optionally stop early if a crash is found
      4) Return the best scenario found and enough info to reproduce.

    Return dict MUST contain at least:
        {
          "best_cfg": Dict[str, Any],
          "best_objectives": Dict[str, Any],
          "best_fitness": float,
          "best_seed_base": int,
          "history": List[float]
        }

    Optional but useful:
        - "best_time_series": ts
        - "evaluations": int
    """
    rng = np.random.default_rng(seed)

    # TODO (students): choose initialization (base_cfg or random scenario)
    current_cfg = dict(base_cfg)
    print("INITIAL CFG:", current_cfg)

    # Evaluate initial solution (seed_base used for reproducibility)
    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    cur_fit = compute_fitness(obj)

    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base

    history = [best_fit]

    best_ts = ts

    # TODO (students): implement HC loop
    # - generate neighbors
    # - evaluate
    # - pick best
    # - accept if improved
    # - early stop on crash (optional)
    evaluations = 0
    # crashes = 0

    for iteration in range(1, iterations):
        # generate neighbors 
        mutated_cfg = best_cfg.copy()
        for neighbor in range(0, neighbors_per_iter):
            seed_base = int(rng.integers(1e9)) # ??? is this needed?
            mutated_cfg = mutate_config(mutated_cfg, param_spec, rng)
            # run the experiment
            crashed, ts = run_episode(env_id, mutated_cfg, policy, defaults, seed_base)
            print("CRASHED", crashed)
            obj = compute_objectives_from_time_series(ts)
            # Crashed is not properly recorded in TS for some reason, we will overwrite the obj function to enforce this.
            if crashed:
                cur_fit = -1
            else:
                cur_fit = compute_fitness(obj)

            # mutated fit is better than best_fit, we choose this.
            if cur_fit <= best_fit:
                print(f"Better Found - New={cur_fit:.4f}, Old={best_fit:.4f}")
                best_fit = cur_fit
                best_obj = obj
                best_seed_base = seed_base
                best_cfg = mutated_cfg
                best_ts = ts
            evaluations += 1
            if crashed:
                # crashes += 1
                record_video_episode(env_id, best_cfg, policy, defaults, best_seed_base, out_dir="videos")
                break
        history.append(best_fit) # log the history of each iteration (consider logging only changes?)
        if best_fit == -1:
        #     # Early stop if we found a crash:
            break
    
    result = {
        "best_cfg" : best_cfg,
        "best_objectives": best_obj,
        "best_fitness": best_fit,
        "best_seed_base": best_seed_base,
        "history" : history,
        # optional
        # "best_time_series": best_ts,
        "evaluations": evaluations,
        # "crashes" : crashes
    }
    return result