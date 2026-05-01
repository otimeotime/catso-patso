"""Optuna driver for CATSO/PATSO/UCT hyperparameter tuning.

Architecture:
    - One C++ binary per env (`mcts-eval-<env>`) that accepts hyperparams via CLI
      and emits one JSON line per invocation.
    - This script samples hyperparams with Optuna's TPE sampler, invokes the
      binary as a subprocess, parses the JSON, and returns cvar_regret.

Single-fidelity for now (one trial budget per Optuna trial); see README for the
multi-fidelity Hyperband upgrade path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import yaml
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent
ENV_CONFIGS_PATH = CONFIG_DIR / "env_configs.yaml"
SEARCH_SPACES_PATH = CONFIG_DIR / "search_spaces.yaml"
RESULTS_ROOT = CONFIG_DIR / "results"


# ---------------------------------------------------------------- config loaders

@dataclass
class EnvConfig:
    binary: str
    cvar_tau: float
    trial_budget: int
    num_seeds: int
    eval_rollouts: int
    threads: int
    horizon: int
    base_seed: int


def load_env_config(env: str) -> EnvConfig:
    with open(ENV_CONFIGS_PATH) as f:
        all_envs = yaml.safe_load(f)
    if env not in all_envs:
        raise SystemExit(f"env '{env}' not in {ENV_CONFIGS_PATH}; "
                         f"available: {sorted(all_envs)}")
    cfg = all_envs[env]
    return EnvConfig(**cfg)


def load_search_space(algo: str) -> dict[str, dict[str, Any]]:
    with open(SEARCH_SPACES_PATH) as f:
        spaces = yaml.safe_load(f)
    if algo not in spaces:
        raise SystemExit(f"algo '{algo}' not in {SEARCH_SPACES_PATH}; "
                         f"available: {sorted(spaces)}")
    return spaces[algo]


# ---------------------------------------------------------------- param sampling

def sample_params(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sample one hyperparameter dict from the named search space."""
    out: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "int":
            out[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        elif kind == "float":
            out[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif kind == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"unknown param type {kind!r} for {name}")
    return out


# ------------------------------------------------------------- subprocess runner

def build_cmd(binary: Path, algo: str, params: dict[str, Any], cfg: EnvConfig) -> list[str]:
    cmd = [
        str(binary),
        "--algo", algo,
        "--cvar-tau", str(cfg.cvar_tau),
        "--trial-budget", str(cfg.trial_budget),
        "--num-seeds", str(cfg.num_seeds),
        "--base-seed", str(cfg.base_seed),
        "--eval-rollouts", str(cfg.eval_rollouts),
        "--threads", str(cfg.threads),
        "--horizon", str(cfg.horizon),
    ]
    # power_mean is fixed at 1.0 (paper-analysed setting); see search_spaces.yaml.
    if algo == "CATSO":
        cmd += ["--n-atoms", str(params["n_atoms"]),
                "--optimism", str(params["optimism"]),
                "--power-mean", "1.0"]
    elif algo == "PATSO":
        cmd += ["--max-particles", str(params["max_particles"]),
                "--optimism", str(params["optimism"]),
                "--power-mean", "1.0"]
    elif algo == "UCT":
        # UCT uses different knobs; if you want to tune it, define a UCT entry
        # in search_spaces.yaml. Default-construct otherwise.
        if "uct_bias" in params:
            cmd += ["--uct-bias", str(params["uct_bias"])]
        if "uct_epsilon" in params:
            cmd += ["--uct-epsilon", str(params["uct_epsilon"])]
    else:
        raise ValueError(f"unknown algo {algo!r}")
    return cmd


def run_eval(cmd: list[str]) -> dict[str, Any]:
    """Invoke the eval binary; return parsed JSON or raise on failure."""
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"eval binary failed (rc={res.returncode})\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {res.stderr.strip()[-500:]}"
        )
    # Last non-blank stdout line is the JSON; everything else is diagnostics.
    lines = [ln for ln in res.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(
            f"eval binary produced no stdout\n  cmd: {' '.join(cmd)}\n"
            f"  stderr: {res.stderr.strip()[-500:]}"
        )
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"eval binary stdout was not JSON: {lines[-1]!r}\n"
            f"  parse error: {e}"
        )


# ------------------------------------------------------------------- objective

def make_objective(env: str, algo: str, cfg: EnvConfig, space: dict):
    binary = ROOT / cfg.binary
    if not binary.exists():
        raise SystemExit(
            f"binary not found: {binary}\n"
            f"build it first:  make {cfg.binary}"
        )

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, space)
        cmd = build_cmd(binary, algo, params, cfg)
        t0 = time.time()
        metrics = run_eval(cmd)
        wall = time.time() - t0

        # Record everything as user attrs for later inspection.
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("driver_wall_sec", wall)
        for k, v in params.items():
            trial.set_user_attr(f"param_{k}", v)

        regret = metrics.get("cvar_regret")
        if regret is None:
            # Fallback: penalize. Optuna won't reach this normally because the
            # binary always emits a regret unless the recommended action's CVaR
            # is unknown to the oracle (shouldn't happen for these envs).
            return float("inf")
        return float(regret)

    return objective


# -------------------------------------------------------------------- artifacts

def write_best_json(study: optuna.Study, out_path: Path) -> None:
    if study.best_trial is None:
        return
    bt = study.best_trial
    out_path.write_text(json.dumps({
        "best_value": bt.value,
        "params": bt.params,
        "user_attrs": bt.user_attrs,
        "trial_number": bt.number,
        "datetime_complete": str(bt.datetime_complete),
    }, indent=2, default=str))


def write_trials_csv(study: optuna.Study, out_path: Path) -> None:
    rows = []
    param_keys: set[str] = set()
    attr_keys: set[str] = set()
    for t in study.trials:
        param_keys.update(t.params.keys())
        attr_keys.update(t.user_attrs.keys())
    param_cols = sorted(param_keys)
    attr_cols = sorted(attr_keys)
    header = ["trial", "value", "state"] + \
             [f"p_{k}" for k in param_cols] + \
             [f"u_{k}" for k in attr_cols]
    for t in study.trials:
        row = [t.number, t.value, t.state.name]
        row += [t.params.get(k) for k in param_cols]
        row += [t.user_attrs.get(k) for k in attr_cols]
        rows.append(row)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# -------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="env id (must be in env_configs.yaml)")
    parser.add_argument("--algo", required=True, choices=["UCT", "CATSO", "PATSO"])
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", default=None,
                        help="Optuna study name. Default: <algo>_<env>_v1")
    parser.add_argument("--best-only", action="store_true",
                        help="Skip optimization; print best params from existing study and exit.")
    parser.add_argument("--seed", type=int, default=12345,
                        help="TPE sampler seed (for reproducibility of the search itself).")
    args = parser.parse_args()

    cfg = load_env_config(args.env)
    space = load_search_space(args.algo)
    study_name = args.study_name or f"{args.algo.lower()}_{args.env}_v1"

    out_dir = RESULTS_ROOT / args.env / args.algo
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "study.db"
    storage = f"sqlite:///{db_path}"

    sampler = TPESampler(seed=args.seed, multivariate=True, group=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )

    if args.best_only:
        if study.best_trial is None:
            print("no completed trials in study", file=sys.stderr)
            return 1
        print(json.dumps({
            "study_name": study_name,
            "n_completed": sum(1 for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE),
            "best_value": study.best_trial.value,
            "best_params": study.best_trial.params,
            "best_attrs": study.best_trial.user_attrs,
        }, indent=2, default=str))
        return 0

    objective = make_objective(args.env, args.algo, cfg, space)

    print(f"[tune] env={args.env} algo={args.algo} study={study_name} "
          f"n_trials={args.n_trials} db={db_path}", file=sys.stderr)
    print(f"[tune] env_config={cfg}", file=sys.stderr)
    print(f"[tune] search_space={list(space.keys())}", file=sys.stderr)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    write_best_json(study, out_dir / "best.json")
    write_trials_csv(study, out_dir / "trials.csv")

    if study.best_trial is not None:
        print(f"\n[tune] best cvar_regret = {study.best_trial.value:.6g}", file=sys.stderr)
        print(f"[tune] best params = {study.best_trial.params}", file=sys.stderr)
        print(f"[tune] artifacts: {out_dir}/best.json, {out_dir}/trials.csv, {db_path}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
