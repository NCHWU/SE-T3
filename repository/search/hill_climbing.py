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

import math


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
    crash = 0
    min_distance = float('inf')
    min_same_lane_distance = float('inf')

    for frame in time_series:
        if frame.get('crashed', False):
            crash = 1
        # ego
        ego = frame.get("ego", None)
        if ego is None: continue
        e_x, e_y = ego["pos"]
        # e_speed, e_heading = ego["speed"], ego["heading"]
        # e_length, e_width = ego["length"], ego["width"]
        e_lane = ego["lane_id"]

        # other
        for other in frame.get("others", []):
            o_x, o_y = other["pos"]
            # o_length, o_width = other["length"], other["width"]
            o_lane = other["lane_id"]
            # compare ego with other vehicle
            # euclidean distance
            diff = ((o_x - e_x)**2 + (o_y - e_y) ** 2) ** 0.5
            min_distance = min(min_distance, diff)
            # compute same-lane difference if it exists
            if e_lane == o_lane:
                same_lane_diff = abs(o_x - e_x)
                min_same_lane_distance = min(min_same_lane_distance, same_lane_diff)

    result = {
        "crash_count" : crash,
        "min_distance" : min_distance,
        "min_same_lane_distance" : min_same_lane_distance
    }
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
        return -1
    same_lane_dist = objectives["min_same_lane_distance"]
    if np.isfinite(same_lane_dist):
        # if min_same_lane_distance exists, then we prefer that over euclidean distance
        fitness = same_lane_dist
        # print("absolute lane distance", fitness)
    else:
        fitness = objectives["min_distance"]
        # print("euclidean distance", fitness)
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
    selections = ['vehicles_count','initial_spacing', 'ego_spacing', 'lanes_count', 'initial_lane_id']
    select = rng.choice(selections)
    m_type = param_spec[select]["type"]
    m_min, m_max = param_spec[select]["min"], param_spec[select]["max"]

    # SPECIAL CASE FOR INITIAL_LANE_ID as it's dependent on lanes_count
    if select == "initial_lane_id":
        # local move between -1 or 1
        delta = int(rng.choice([-1, 1]))
        mutated_config["initial_lane_id"] = int(
            np.clip(mutated_config["initial_lane_id"] + delta, 0, mutated_config["lanes_count"] - 1)
        )
        return mutated_config
    
    if m_type == "int": # mutate vehicles_count and lanes_count
        step_size = 1
        if select == 'vehicles_count':
            # step vehicle_counts by 1-5 steps
            step_size = int(rng.choice([1, 2, 3, 4, 5]))
        # step negatively or positively
        step_direction = rng.choice([-1, 1])
        delta = step_size * step_direction
        mutated_config[select] = int(np.clip(mutated_config[select] + delta, m_min, m_max))
    else:
        # else, we mutate initial_spacing and ego_spacing
        step_range = m_max - m_min
        step_size = float(rng.choice([0.1, 0.15, 0.2, 0.25]))
        sigma = step_size * step_range
        mutated_config[select] = float(np.clip(mutated_config[select] + rng.normal(0, sigma), m_min, m_max))
    
    if select == "lanes_count":
        # everytime we modify lanes_count, we should update initial_lane_id as well
        if "initial_lane_id" not in mutated_config:
            # ensure initial_lane_id is inside mutated_config
            mutated_config["initial_lane_id"] = int(rng.integers(0, mutated_config["lanes_count"]))
        # clip lane id to [0, lanes_count-1] as per spec
        mutated_config["initial_lane_id"] = int(
            np.clip(mutated_config["initial_lane_id"], 0, mutated_config["lanes_count"] - 1)
        )
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
    # INITIALIZE RANDOM SCENARIO
    for k, v in param_spec.items():
        if k not in current_cfg:
            if v["type"] == "int":
                current_cfg[k] = int(rng.integers(v["min"], v["max"] + 1))
            else:
                current_cfg[k] = float(rng.uniform(v["min"], v["max"]))
    if "lanes_count" in current_cfg and "initial_lane_id" in current_cfg:
        current_cfg["initial_lane_id"] = int(np.clip(current_cfg["initial_lane_id"], 0, current_cfg["lanes_count"] - 1))

    # Evaluate initial solution (seed_base used for reproducibility)
    seed_base = int(rng.integers(1e9))
    print(seed_base)
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
        print("Iteration 1")
        # generate neighbors 
        seed_base = int(rng.integers(1e9)) # ??? is this needed?
        for neighbor in range(1, neighbors_per_iter + 1):
            print("Neighbor", neighbor)
            mutated_cfg = mutate_config(best_cfg, param_spec, rng)
            # run the experiment
            crashed, ts = run_episode(env_id, mutated_cfg, policy, defaults, seed_base)
            # print("CRASHED", crashed)
            obj = compute_objectives_from_time_series(ts)
            # Crashed is not properly recorded in TS for some reason, we will overwrite the obj function to enforce this.
            if crashed:
                obj["crash_count"] = 1
            cur_fit = compute_fitness(obj)

            # mutated fit is better than best_fit, we choose this.
            if cur_fit < best_fit:
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
        # Early stop if we found a crash:
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