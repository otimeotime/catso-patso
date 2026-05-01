"""Run a multi-budget comparison using tuned hyperparameters.

For each (algo, trial_budget) pair, invokes the C++ `mcts-eval-<env>` binary
with the *tuned* hyperparams loaded from `tune/results/<env>/<algo>/best.json`
(or built-in defaults for UCT, since UCT is the untuned baseline). Writes a
summary CSV in the schema `plot.py` already understands, so a single
`python3 plot.py results_<env>_summary.csv` produces the comparison plots.

Example:
    python3 tune/compare.py --env bettinggame \
        --algos UCT CATSO PATSO \
        --trial-budgets 1000 2000 5000 10000 25000 \
        --num-seeds 5 \
        --out results_bettinggame_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent
ENV_CONFIGS_PATH = CONFIG_DIR / "env_configs.yaml"
RESULTS_ROOT = CONFIG_DIR / "results"

# Default UCT settings (matches discrete_runner_common.h's build_candidates).
UCT_DEFAULTS: dict[str, Any] = {"uct_bias": -1.0, "uct_epsilon": 0.1}


def load_env_config(env: str) -> dict[str, Any]:
    with open(ENV_CONFIGS_PATH) as f:
        all_envs = yaml.safe_load(f)
    if env not in all_envs:
        raise SystemExit(
            f"env '{env}' not in {ENV_CONFIGS_PATH}; available: {sorted(all_envs)}"
        )
    return all_envs[env]


def load_best_params(env: str, algo: str) -> dict[str, Any] | None:
    """Return the tuned hyperparams for (env, algo), or None if no best.json yet."""
    p = RESULTS_ROOT / env / algo / "best.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["params"]


def build_cmd(
    binary: Path,
    algo: str,
    params: dict[str, Any],
    env_cfg: dict[str, Any],
    trial_budget: int,
    num_seeds: int,
) -> list[str]:
    cmd = [
        str(binary),
        "--algo", algo,
        "--cvar-tau", str(env_cfg["cvar_tau"]),
        "--trial-budget", str(trial_budget),
        "--num-seeds", str(num_seeds),
        "--base-seed", str(env_cfg["base_seed"]),
        "--eval-rollouts", str(env_cfg["eval_rollouts"]),
        "--threads", str(env_cfg["threads"]),
        "--horizon", str(env_cfg["horizon"]),
    ]
    if algo == "CATSO":
        cmd += ["--n-atoms", str(params["n_atoms"]),
                "--optimism", str(params["optimism"]),
                "--power-mean", "1.0"]  # fixed
    elif algo == "PATSO":
        cmd += ["--max-particles", str(params["max_particles"]),
                "--optimism", str(params["optimism"]),
                "--power-mean", "1.0"]
    elif algo == "UCT":
        cmd += ["--uct-bias", str(params.get("uct_bias", UCT_DEFAULTS["uct_bias"])),
                "--uct-epsilon", str(params.get("uct_epsilon", UCT_DEFAULTS["uct_epsilon"]))]
    else:
        raise ValueError(f"unknown algo {algo!r}")
    return cmd


def run_eval(cmd: list[str]) -> dict[str, Any]:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"eval binary failed (rc={res.returncode})\n  cmd: {' '.join(cmd)}\n"
            f"  stderr: {res.stderr.strip()[-500:]}"
        )
    lines = [ln for ln in res.stdout.strip().splitlines() if ln.strip()]
    return json.loads(lines[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", required=True)
    ap.add_argument("--algos", nargs="+", default=["UCT", "CATSO", "PATSO"])
    ap.add_argument("--trial-budgets", nargs="+", type=int,
                    default=[1000, 2000, 5000, 10000, 25000])
    ap.add_argument("--num-seeds", type=int, default=5,
                    help="seeds per (algo, budget) cell. 5 is paper-grade; 3 is fine for iteration.")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: results_<env>_summary.csv at repo root)")
    args = ap.parse_args()

    env_cfg = load_env_config(args.env)
    binary = ROOT / env_cfg["binary"]
    if not binary.exists():
        raise SystemExit(f"binary not found: {binary}\nbuild first:  make {env_cfg['binary']}")

    out_path = Path(args.out) if args.out else (ROOT / f"results_{args.env}_summary.csv")

    # Resolve tuned params per algo.
    params_by_algo: dict[str, dict[str, Any]] = {}
    for algo in args.algos:
        if algo == "UCT":
            params_by_algo[algo] = UCT_DEFAULTS
            print(f"[compare] {algo}: using defaults {UCT_DEFAULTS}", file=sys.stderr)
            continue
        best = load_best_params(args.env, algo)
        if best is None:
            print(f"[compare] WARNING: no tuned best.json for {algo} on {args.env}; "
                  f"run `python3 tune/tune.py --env {args.env} --algo {algo}` first.",
                  file=sys.stderr)
            continue
        params_by_algo[algo] = best
        print(f"[compare] {algo}: tuned params {best}", file=sys.stderr)

    if not params_by_algo:
        raise SystemExit("no algos to compare (no tuned configs found and UCT not requested)")

    # Schema matches plot.py's expectation. mc_cvar_stddev is optional; we include
    # it because eval_<env> emits it.
    fieldnames = [
        "env", "algorithm", "trial",
        "mc_mean", "mc_stddev", "mc_cvar", "mc_cvar_stddev",
        "cvar_regret", "optimal_action_prob", "catastrophic_count",
    ]
    rows = []
    total = len(params_by_algo) * len(args.trial_budgets)
    done = 0
    t_start = time.time()

    for algo, params in params_by_algo.items():
        for budget in args.trial_budgets:
            cmd = build_cmd(binary, algo, params, env_cfg, budget, args.num_seeds)
            t0 = time.time()
            metrics = run_eval(cmd)
            wall = time.time() - t0
            done += 1
            print(f"[compare] {done}/{total} {algo} @ {budget} trials: "
                  f"regret={metrics['cvar_regret']:.4g} mc_cvar={metrics['mc_cvar']:.4g} "
                  f"({wall:.1f}s)", file=sys.stderr)
            rows.append({
                "env": args.env,
                "algorithm": algo,
                "trial": budget,
                "mc_mean": metrics["mc_mean"],
                "mc_stddev": metrics["mc_stddev"],
                "mc_cvar": metrics["mc_cvar"],
                "mc_cvar_stddev": metrics.get("cvar_regret_stddev", 0.0),
                "cvar_regret": metrics["cvar_regret"],
                "optimal_action_prob": metrics["optimal_action_prob"],
                "catastrophic_count": metrics["catastrophic_count"],
            })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[compare] wrote {len(rows)} rows to {out_path} "
          f"(total wall {time.time() - t_start:.1f}s)", file=sys.stderr)
    print(f"[compare] now plot:  python3 plot.py {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
