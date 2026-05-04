#include "exp/tuning_runner.h"

#include "exp/evaluation_utils.h"

#include "mcts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace mcts::exp {
    namespace {
        struct SummaryAccumulator {
            int count = 0;
            int regret_count = 0;
            int hit_count = 0;

            double sum_mc_mean = 0.0;
            double sum_mc_stddev = 0.0;
            double sum_mc_cvar = 0.0;
            double sum_mc_cvar_stddev = 0.0;
            double sum_cvar_regret = 0.0;
            double sum_optimal_action_hit = 0.0;
            double sum_catastrophic_count = 0.0;
        };

        std::string csv_escape(const std::string& value) {
            if (value.find_first_of(",\"\n\r") == std::string::npos) {
                return value;
            }

            std::string escaped;
            escaped.reserve(value.size() + 2);
            escaped.push_back('"');
            for (char ch : value) {
                if (ch == '"') {
                    escaped += "\"\"";
                }
                else {
                    escaped.push_back(ch);
                }
            }
            escaped.push_back('"');
            return escaped;
        }

        void run_trials_with_progress(
            mcts::MctsPool& pool,
            int total_trials,
            int batch_trials,
            const std::string& label)
        {
            const int actual_batch_trials = std::max(1, std::min(batch_trials, total_trials));
            int completed_trials = 0;
            auto start_time = std::chrono::steady_clock::now();

            while (completed_trials < total_trials) {
                const int trials_to_add = std::min(actual_batch_trials, total_trials - completed_trials);
                pool.run_trials(trials_to_add, std::numeric_limits<double>::max(), true);
                completed_trials += trials_to_add;

                const double progress =
                    100.0 * static_cast<double>(completed_trials) / static_cast<double>(total_trials);
                const double elapsed_seconds =
                    std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::steady_clock::now() - start_time).count();

                std::ostringstream oss;
                oss << "    " << label
                    << " trials " << completed_trials << "/" << total_trials
                    << " (" << std::fixed << std::setprecision(1) << progress << "%, "
                    << std::setprecision(2) << elapsed_seconds << "s)";
                std::cout << oss.str() << "\n";
            }
        }

        struct RankedSummary {
            double score = 0.0;
            double tie_break = 0.0;
            bool uses_regret = false;
        };

        RankedSummary rank_summary(const SummaryAccumulator& summary) {
            if (summary.regret_count > 0) {
                const double mean_regret = summary.sum_cvar_regret / static_cast<double>(summary.regret_count);
                const double mean_hit = (summary.hit_count > 0)
                    ? (summary.sum_optimal_action_hit / static_cast<double>(summary.hit_count))
                    : -std::numeric_limits<double>::infinity();
                return {mean_regret, -mean_hit, true};
            }

            const double mean_mc_cvar = summary.sum_mc_cvar / static_cast<double>(summary.count);
            const double mean_mc_mean = summary.sum_mc_mean / static_cast<double>(summary.count);
            return {-mean_mc_cvar, -mean_mc_mean, false};
        }
    }

    int run_tuning_experiment(const ExperimentSpec& spec, const std::vector<Candidate>& candidates) {
        if (!spec.make_env) {
            throw std::runtime_error("ExperimentSpec.make_env must be set");
        }

        auto env = spec.make_env();
        auto init_state = env->get_initial_state_itfc();
        const std::string tune_csv = spec.tune_output_csv.empty()
            ? ("tune_" + spec.env_name + ".csv")
            : spec.tune_output_csv;
        const int tune_threads = (spec.tune_threads > 0) ? spec.tune_threads : spec.threads;
        const int progress_batch_trials = std::max(1, std::min(spec.progress_batch_trials, spec.tune_total_trials));

        std::ofstream out(tune_csv, std::ios::out | std::ios::trunc);
        out << std::setprecision(17);
        out << "env,algorithm,config,eval_tau,run,"
            << "mc_mean,mc_stddev,mc_cvar,mc_cvar_stddev,"
            << "cvar_regret,optimal_action_hit,catastrophic_count\n";

        std::map<std::string, std::map<double, std::unordered_map<std::string, SummaryAccumulator>>> summary;

        if (spec.print_env_info) {
            spec.print_env_info(std::cout);
        }
        else {
            std::cout << "[tune] " << spec.display_name << "\n";
        }
        std::cout << "  Discount gamma: " << spec.discount_gamma << "\n";

        std::cout << "  Candidates: " << candidates.size()
                  << ", Runs: " << spec.tune_runs
                  << ", Trials per search: " << spec.tune_total_trials
                  << ", Threads: " << tune_threads
                  << "\n";

        const int run_summary_stride = std::max(1, spec.tune_runs / 10);
        for (size_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
            const auto& cand = candidates[cand_idx];
            std::cout << "[" << (cand_idx + 1) << "/" << candidates.size() << "] "
                      << cand.algo << " cfg=" << cand.config
                      << ", eval_tau=" << cand.eval_tau << "\n";

            for (int run = 0; run < spec.tune_runs; ++run) {
                const int seed = spec.base_seed
                    + 1000 * run
                    + static_cast<int>(std::hash<std::string>{}(cand.algo + cand.config) % 997);

                auto [mgr, root] = cand.make(
                    std::static_pointer_cast<mcts::MctsEnv>(env),
                    init_state,
                    spec.horizon,
                    seed);
                auto pool = std::make_unique<mcts::MctsPool>(mgr, root, tune_threads);

                if (run == 0) {
                    std::ostringstream label;
                    label << "[" << (cand_idx + 1) << "/" << candidates.size() << "] "
                          << cand.algo << " run " << (run + 1) << "/" << spec.tune_runs;
                    run_trials_with_progress(*pool, spec.tune_total_trials, progress_batch_trials, label.str());
                }
                else {
                    pool->run_trials(spec.tune_total_trials, std::numeric_limits<double>::max(), true);
                }

                const EvalStats stats = evaluate_tree_generic(
                    std::static_pointer_cast<const mcts::MctsEnv>(env),
                    std::static_pointer_cast<const mcts::MctsDNode>(root),
                    spec.horizon,
                    spec.eval_rollouts,
                    tune_threads,
                    seed + 777,
                    cand.eval_tau,
                    spec.is_catastrophic_transition);

                RootMetrics metrics;
                if (spec.evaluate_root_metrics) {
                    metrics = spec.evaluate_root_metrics(
                        std::static_pointer_cast<const mcts::MctsEnv>(env),
                        std::static_pointer_cast<const mcts::MctsDNode>(root),
                        cand.eval_tau,
                        spec.discount_gamma);
                }

                out << spec.env_name << "," << cand.algo << "," << csv_escape(cand.config)
                    << "," << cand.eval_tau << "," << run
                    << "," << stats.mean
                    << "," << stats.stddev
                    << "," << stats.cvar
                    << "," << stats.cvar_stddev
                    << "," << metrics.cvar_regret
                    << "," << metrics.optimal_action_hit
                    << "," << stats.catastrophic_count
                    << "\n";

                SummaryAccumulator& agg = summary[cand.algo][cand.eval_tau][cand.config];
                agg.count++;
                agg.sum_mc_mean += stats.mean;
                agg.sum_mc_stddev += stats.stddev;
                agg.sum_mc_cvar += stats.cvar;
                agg.sum_mc_cvar_stddev += stats.cvar_stddev;
                agg.sum_catastrophic_count += static_cast<double>(stats.catastrophic_count);
                if (!std::isnan(metrics.cvar_regret)) {
                    agg.regret_count++;
                    agg.sum_cvar_regret += metrics.cvar_regret;
                }
                if (!std::isnan(metrics.optimal_action_hit)) {
                    agg.hit_count++;
                    agg.sum_optimal_action_hit += metrics.optimal_action_hit;
                }

                if (run == 0 || run + 1 == spec.tune_runs || ((run + 1) % run_summary_stride == 0)) {
                    std::cout << "    completed run " << (run + 1) << "/" << spec.tune_runs
                              << " regret=" << metrics.cvar_regret
                              << " hit=" << metrics.optimal_action_hit
                              << " mean=" << stats.mean
                              << "\n";
                }
            }

            std::cout << "  done " << cand.algo << " cfg=" << cand.config << "\n";
        }

        std::cout << "\nBest configs (avg over runs per config, grouped by eval_tau):\n";
        std::cout << std::fixed << std::setprecision(6);
        for (const auto& [algo, tau_map] : summary) {
            for (const auto& [eval_tau, cfg_map] : tau_map) {
                RankedSummary best_rank{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), false};
                std::string best_cfg;

                for (const auto& [cfg, agg] : cfg_map) {
                    const RankedSummary current_rank = rank_summary(agg);
                    const bool replace =
                        best_cfg.empty()
                        || (current_rank.uses_regret && !best_rank.uses_regret)
                        || (current_rank.uses_regret == best_rank.uses_regret
                            && (current_rank.score < best_rank.score - 1e-15
                                || (std::abs(current_rank.score - best_rank.score) <= 1e-15
                                    && current_rank.tie_break < best_rank.tie_break - 1e-15)));
                    if (replace) {
                        best_rank = current_rank;
                        best_cfg = cfg;
                    }
                }

                if (best_cfg.empty()) {
                    continue;
                }

                const SummaryAccumulator& agg = cfg_map.at(best_cfg);
                std::cout << "  " << algo
                          << " (eval_tau=" << eval_tau << ")";
                if (agg.regret_count > 0) {
                    const double avg_regret = agg.sum_cvar_regret / static_cast<double>(agg.regret_count);
                    const double avg_hit = (agg.hit_count > 0)
                        ? (agg.sum_optimal_action_hit / static_cast<double>(agg.hit_count))
                        : std::numeric_limits<double>::quiet_NaN();
                    std::cout << " -> avg_regret=" << avg_regret
                              << ", avg_hit=" << avg_hit;
                }
                else {
                    std::cout << " -> avg_mc_cvar="
                              << (agg.sum_mc_cvar / static_cast<double>(agg.count));
                }
                std::cout << ", avg_mean=" << (agg.sum_mc_mean / static_cast<double>(agg.count))
                          << ", cfg=" << best_cfg
                          << "\n";
            }
        }

        std::cout << "Results written to " << tune_csv << "\n";
        return 0;
    }
}
