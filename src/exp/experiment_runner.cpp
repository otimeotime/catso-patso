#include "exp/experiment_runner.h"

#include "exp/evaluation_utils.h"
#include "exp/manager_config_printer.h"

#include "mcts.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
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
    }

    int run_experiment(const ExperimentSpec& spec, const std::vector<Candidate>& candidates) {
        if (!spec.make_env) {
            throw std::runtime_error("ExperimentSpec.make_env must be set");
        }

        auto env = spec.make_env();
        auto init_state = env->get_initial_state_itfc();
        std::map<std::pair<std::string, int>, SummaryAccumulator> summary_by_algo_and_trial;

        const std::string raw_csv = spec.output_prefix + ".csv";
        const std::string summary_csv = spec.output_prefix + "_summary.csv";

        std::ofstream out(raw_csv, std::ios::out | std::ios::trunc);
        out << std::setprecision(17);
        out << "env,algorithm,run,trial,"
            << "mc_mean,mc_stddev,mc_cvar,mc_cvar_stddev,"
            << "cvar_regret,optimal_action_hit,catastrophic_count\n";

        if (spec.print_env_info) {
            spec.print_env_info(std::cout);
        }
        else {
            std::cout << "[exp] " << spec.display_name << "\n";
        }

        std::cout << "  Algorithms: " << candidates.size()
                  << ", Runs: " << spec.runs
                  << ", Trial counts: " << spec.trial_counts.size()
                  << "\n";

        for (const auto& cand : candidates) {
            bool printed_config = false;

            for (int run = 0; run < spec.runs; ++run) {
                const int seed = spec.base_seed
                    + 1000 * run
                    + static_cast<int>(std::hash<std::string>{}(cand.algo) % 997);

                auto [mgr, root] = cand.make(
                    std::static_pointer_cast<mcts::MctsEnv>(env),
                    init_state,
                    spec.horizon,
                    seed);

                if (!printed_config) {
                    const std::string runtime_config =
                        describe_manager_config(std::static_pointer_cast<const mcts::MctsManager>(mgr));
                    std::cout << "\nRunning " << cand.algo;
                    if (!runtime_config.empty()) {
                        std::cout << " [" << runtime_config << "]";
                    }
                    else if (!cand.config.empty()) {
                        std::cout << " [" << cand.config << "]";
                    }
                    std::cout << "...\n";
                    printed_config = true;
                }

                auto pool = std::make_unique<mcts::MctsPool>(mgr, root, spec.threads);

                int trials_so_far = 0;
                for (int target_trials : spec.trial_counts) {
                    const int trials_to_add = target_trials - trials_so_far;
                    if (trials_to_add > 0) {
                        pool->run_trials(trials_to_add, std::numeric_limits<double>::max(), true);
                        trials_so_far = target_trials;
                    }

                    const EvalStats stats = evaluate_tree_generic(
                        std::static_pointer_cast<const mcts::MctsEnv>(env),
                        std::static_pointer_cast<const mcts::MctsDNode>(root),
                        spec.horizon,
                        spec.eval_rollouts,
                        spec.threads,
                        seed + 777,
                        cand.eval_tau,
                        spec.is_catastrophic_transition);

                    RootMetrics metrics;
                    if (spec.evaluate_root_metrics) {
                        metrics = spec.evaluate_root_metrics(
                            std::static_pointer_cast<const mcts::MctsEnv>(env),
                            std::static_pointer_cast<const mcts::MctsDNode>(root),
                            cand.eval_tau);
                    }

                    out << spec.env_name << "," << cand.algo << "," << run
                        << "," << target_trials
                        << "," << stats.mean
                        << "," << stats.stddev
                        << "," << stats.cvar
                        << "," << stats.cvar_stddev
                        << "," << metrics.cvar_regret
                        << "," << metrics.optimal_action_hit
                        << "," << stats.catastrophic_count
                        << "\n";

                    SummaryAccumulator& summary = summary_by_algo_and_trial[{cand.algo, target_trials}];
                    summary.count++;
                    summary.sum_mc_mean += stats.mean;
                    summary.sum_mc_stddev += stats.stddev;
                    summary.sum_mc_cvar += stats.cvar;
                    summary.sum_mc_cvar_stddev += stats.cvar_stddev;
                    summary.sum_catastrophic_count += static_cast<double>(stats.catastrophic_count);

                    if (!std::isnan(metrics.cvar_regret)) {
                        summary.regret_count++;
                        summary.sum_cvar_regret += metrics.cvar_regret;
                    }
                    if (!std::isnan(metrics.optimal_action_hit)) {
                        summary.hit_count++;
                        summary.sum_optimal_action_hit += metrics.optimal_action_hit;
                    }
                }

                std::cout << "  " << cand.algo << " run " << (run + 1) << "/" << spec.runs << " done\n";
            }
        }

        std::ofstream summary_out(summary_csv, std::ios::out | std::ios::trunc);
        summary_out << std::setprecision(17);
        summary_out << "env,algorithm,trial,"
                    << "mc_mean,mc_stddev,mc_cvar,mc_cvar_stddev,"
                    << "cvar_regret,optimal_action_prob,catastrophic_count\n";

        for (const auto& [algo_trial, summary] : summary_by_algo_and_trial) {
            const std::string& algo = algo_trial.first;
            const int trial = algo_trial.second;
            const double mean_cvar_regret =
                (summary.regret_count > 0)
                ? (summary.sum_cvar_regret / static_cast<double>(summary.regret_count))
                : std::numeric_limits<double>::quiet_NaN();
            const double mean_optimal_action_hit =
                (summary.hit_count > 0)
                ? (summary.sum_optimal_action_hit / static_cast<double>(summary.hit_count))
                : std::numeric_limits<double>::quiet_NaN();

            summary_out << spec.env_name << "," << csv_escape(algo) << "," << trial
                        << "," << (summary.sum_mc_mean / static_cast<double>(summary.count))
                        << "," << (summary.sum_mc_stddev / static_cast<double>(summary.count))
                        << "," << (summary.sum_mc_cvar / static_cast<double>(summary.count))
                        << "," << (summary.sum_mc_cvar_stddev / static_cast<double>(summary.count))
                        << "," << mean_cvar_regret
                        << "," << mean_optimal_action_hit
                        << "," << (summary.sum_catastrophic_count / static_cast<double>(summary.count))
                        << "\n";
        }

        std::cout << "\nResults written to " << raw_csv << " and " << summary_csv << "\n";
        return 0;
    }
}
