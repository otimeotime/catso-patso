#pragma once

#include "algorithms/catso/catso_chance_node.h"
#include "algorithms/catso/catso_decision_node.h"
#include "algorithms/catso/catso_manager.h"
#include "algorithms/catso/patso_decision_node.h"
#include "algorithms/catso/patso_manager.h"
#include "algorithms/uct/uct_chance_node.h"
#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/uct_manager.h"

#include "exp/manager_config_printer.h"
#include "mc_eval.h"
#include "mcts.h"
#include "mcts_env_context.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mcts::exp::runner {
    constexpr double kCvarTolerance = 1e-12;

    struct Candidate {
        std::string algo;
        std::string config;
        std::function<std::pair<std::shared_ptr<mcts::MctsManager>, std::shared_ptr<mcts::MctsDNode>>(
            std::shared_ptr<mcts::MctsEnv> env,
            std::shared_ptr<const mcts::State> init_state,
            int max_depth,
            int seed)> make;
    };

    struct EvalStats {
        double mean;
        double stddev;
        double cvar;
        int catastrophic_count;
    };

    using ReturnDistribution = std::map<double, double>;

    struct OptimalDistributionSolution {
        ReturnDistribution state_value_distribution;
        std::map<int, ReturnDistribution> action_value_distributions;
        std::map<int, double> action_cvars;
        std::vector<int> optimal_actions;
        int canonical_action = -1;
        double optimal_cvar = 0.0;
    };

    struct ExperimentMetrics {
        double cvar_regret;
        double optimal_action_hit;
    };

    struct SummaryAccumulator {
        int count = 0;
        int regret_count = 0;
        double sum_mc_mean = 0.0;
        double sum_mc_stddev = 0.0;
        double sum_mc_cvar = 0.0;
        double sum_cvar_regret = 0.0;
        double sum_optimal_action_hit = 0.0;
        double sum_catastrophic_count = 0.0;
    };

    struct SimpleEvalStats {
        double cvar_return;
        double cvar_stddev;
    };

    struct SimpleSummaryAccumulator {
        int count = 0;
        double sum_cvar_return = 0.0;
        double sum_cvar_stddev = 0.0;
    };

    template <typename CatastropheFn>
    requires requires (const CatastropheFn& fn, std::shared_ptr<const mcts::State> s) {
        fn(s);
    }
    bool catastrophe_from_initial_state(
        const CatastropheFn& catastrophe_fn,
        std::shared_ptr<const mcts::State> state)
    {
        return catastrophe_fn(state);
    }

    template <typename CatastropheFn>
    requires (!requires (const CatastropheFn& fn, std::shared_ptr<const mcts::State> s) {
        fn(s);
    })
    bool catastrophe_from_initial_state(
        const CatastropheFn& catastrophe_fn,
        std::shared_ptr<const mcts::State> state)
    {
        (void)catastrophe_fn;
        (void)state;
        return false;
    }

    template <typename CatastropheFn>
    requires requires (
        const CatastropheFn& fn,
        std::shared_ptr<const mcts::State> s,
        std::shared_ptr<const mcts::Action> a,
        std::shared_ptr<const mcts::State> ns) {
        fn(s, a, ns);
    }
    bool catastrophe_from_transition(
        const CatastropheFn& catastrophe_fn,
        std::shared_ptr<const mcts::State> state,
        std::shared_ptr<const mcts::Action> action,
        std::shared_ptr<const mcts::State> next_state)
    {
        return catastrophe_fn(state, action, next_state);
    }

    template <typename CatastropheFn>
    requires (!requires (
        const CatastropheFn& fn,
        std::shared_ptr<const mcts::State> s,
        std::shared_ptr<const mcts::Action> a,
        std::shared_ptr<const mcts::State> ns) {
        fn(s, a, ns);
    })
    bool catastrophe_from_transition(
        const CatastropheFn& catastrophe_fn,
        std::shared_ptr<const mcts::State> state,
        std::shared_ptr<const mcts::Action> action,
        std::shared_ptr<const mcts::State> next_state)
    {
        (void)state;
        (void)action;
        return catastrophe_fn(next_state);
    }

    inline double compute_lower_tail_cvar(const ReturnDistribution& distribution, double tau) {
        if (distribution.empty()) {
            return 0.0;
        }

        const double tau_clamped = std::max(kCvarTolerance, std::min(tau, 1.0));
        double total_mass = 0.0;
        for (const auto& [value, probability] : distribution) {
            (void)value;
            if (probability > 0.0) {
                total_mass += probability;
            }
        }

        if (total_mass <= 0.0) {
            return 0.0;
        }

        const double tail_mass = tau_clamped * total_mass;
        double cumulative_mass = 0.0;
        double tail_sum = 0.0;

        for (const auto& [value, probability] : distribution) {
            if (probability <= 0.0) {
                continue;
            }

            const double remaining_mass = std::max(0.0, tail_mass - cumulative_mass);
            const double mass_to_take = std::min(probability, remaining_mass);
            tail_sum += mass_to_take * value;
            cumulative_mass += probability;

            if (cumulative_mass >= tail_mass) {
                break;
            }
        }

        return tail_sum / tail_mass;
    }

    inline double compute_empirical_lower_tail_cvar(std::vector<double> samples, double tau) {
        if (samples.empty()) {
            return 0.0;
        }

        std::sort(samples.begin(), samples.end());

        const double tau_clamped = std::max(kCvarTolerance, std::min(tau, 1.0));
        const double tail_mass = tau_clamped * static_cast<double>(samples.size());
        double cumulative_mass = 0.0;
        double tail_sum = 0.0;

        for (double sample : samples) {
            const double remaining_mass = std::max(0.0, tail_mass - cumulative_mass);
            const double mass_to_take = std::min(1.0, remaining_mass);
            tail_sum += mass_to_take * sample;
            cumulative_mass += 1.0;

            if (cumulative_mass >= tail_mass) {
                break;
            }
        }

        return tail_sum / tail_mass;
    }

    template <typename StateT>
    using StateKey = std::decay_t<decltype(std::declval<StateT>().state)>;

    template <typename EnvT, typename StateT>
    class CvarOracle {
        private:
            std::shared_ptr<const EnvT> env;
            double tau;
            std::map<StateKey<StateT>, OptimalDistributionSolution> memo;

            static StateKey<StateT> make_key(std::shared_ptr<const StateT> state) {
                return state->state;
            }

            static void add_weighted_shifted_distribution(
                ReturnDistribution& out,
                const ReturnDistribution& in,
                double weight,
                double shift)
            {
                for (const auto& [value, probability] : in) {
                    out[value + shift] += weight * probability;
                }
            }

        public:
            CvarOracle(std::shared_ptr<const EnvT> env, double tau) : env(std::move(env)), tau(tau) {}

            const OptimalDistributionSolution& solve_state(std::shared_ptr<const StateT> state) {
                const StateKey<StateT> key = make_key(state);
                const auto memo_it = memo.find(key);
                if (memo_it != memo.end()) {
                    return memo_it->second;
                }

                OptimalDistributionSolution solution;
                if (env->is_sink_state(state)) {
                    solution.state_value_distribution[0.0] = 1.0;
                    auto [it, inserted] = memo.emplace(key, std::move(solution));
                    (void)inserted;
                    return it->second;
                }

                auto actions = env->get_valid_actions(state);
                for (const auto& action : *actions) {
                    ReturnDistribution action_distribution;
                    auto transitions = env->get_transition_distribution(state, action);
                    for (const auto& [next_state, probability] : *transitions) {
                        const auto& child_solution = solve_state(next_state);
                        const double local_reward = env->get_reward(state, action, next_state);
                        add_weighted_shifted_distribution(
                            action_distribution,
                            child_solution.state_value_distribution,
                            probability,
                            local_reward);
                    }

                    const int action_id = action->action;
                    const double action_cvar = compute_lower_tail_cvar(action_distribution, tau);
                    solution.action_value_distributions.emplace(action_id, action_distribution);
                    solution.action_cvars.emplace(action_id, action_cvar);

                    if (solution.optimal_actions.empty() || action_cvar > solution.optimal_cvar + kCvarTolerance) {
                        solution.optimal_cvar = action_cvar;
                        solution.optimal_actions = {action_id};
                        solution.canonical_action = action_id;
                    }
                    else if (std::abs(action_cvar - solution.optimal_cvar) <= kCvarTolerance) {
                        solution.optimal_actions.push_back(action_id);
                        solution.canonical_action = std::min(solution.canonical_action, action_id);
                    }
                }

                if (solution.canonical_action >= 0) {
                    solution.state_value_distribution = solution.action_value_distributions.at(solution.canonical_action);
                }

                auto [it, inserted] = memo.emplace(key, std::move(solution));
                (void)inserted;
                return it->second;
            }
    };

    template <typename EnvT, typename CatastropheFn>
    EvalStats evaluate_tree(
        std::shared_ptr<const EnvT> env,
        std::shared_ptr<const mcts::MctsDNode> root,
        int horizon,
        int rollouts,
        int threads,
        int seed,
        double cvar_tau,
        CatastropheFn catastrophe_fn,
        bool debug_trajectories = false,
        const std::string& debug_trajectory_label = "")
    {
        (void)threads;

        mcts::RandManager eval_rng(seed);
        mcts::EvalPolicy policy(root, env, eval_rng);
        std::vector<double> sampled_returns;
        sampled_returns.reserve(static_cast<size_t>(std::max(rollouts, 0)));
        int catastrophic_count = 0;
        const bool should_debug_trajectories =
            debug_trajectories || mcts::should_debug_mc_eval_trajectories_from_env();
        const std::string trajectory_label = debug_trajectory_label.empty() ? "MC" : debug_trajectory_label;

        for (int rollout = 0; rollout < rollouts; ++rollout) {
            policy.reset();

            int num_actions_taken = 0;
            double sample_return = 0.0;
            std::shared_ptr<const mcts::State> state = env->get_initial_state_itfc();
            bool rollout_was_catastrophic = catastrophe_from_initial_state(catastrophe_fn, state);
            auto context_ptr = env->sample_context_itfc(state);
            mcts::MctsEnvContext& context = *context_ptr;
            mcts::RolloutTrajectoryTrace trajectory_trace(should_debug_trajectories, trajectory_label);
            trajectory_trace.append_state(state);

            while (num_actions_taken < horizon && !env->is_sink_state_itfc(state)) {
                std::shared_ptr<const mcts::Action> action = policy.get_action(state, context);
                std::shared_ptr<const mcts::State> next_state =
                    env->sample_transition_distribution_itfc(state, action, eval_rng);
                std::shared_ptr<const mcts::Observation> observation =
                    env->sample_observation_distribution_itfc(action, next_state, eval_rng);

                const double reward = env->get_reward_itfc(state, action, observation);
                sample_return += reward;
                trajectory_trace.append_reward(reward);
                trajectory_trace.append_state(next_state);
                if (catastrophe_from_transition(catastrophe_fn, state, action, next_state)) {
                    rollout_was_catastrophic = true;
                }
                policy.update_step(action, observation);
                state = next_state;
                num_actions_taken++;
            }

            trajectory_trace.print(std::cerr);
            sampled_returns.push_back(sample_return);
            if (rollout_was_catastrophic) {
                catastrophic_count++;
            }
        }

        if (sampled_returns.empty()) {
            return {0.0, 0.0, 0};
        }

        double mean = 0.0;
        for (double sample_return : sampled_returns) {
            mean += sample_return;
        }
        mean /= static_cast<double>(sampled_returns.size());

        double stddev = 0.0;
        if (sampled_returns.size() > 1u) {
            double variance = 0.0;
            for (double sample_return : sampled_returns) {
                variance += std::pow(sample_return - mean, 2.0);
            }
            variance /= static_cast<double>(sampled_returns.size() - 1u);
            stddev = std::sqrt(variance);
        }

        const double cvar = compute_empirical_lower_tail_cvar(sampled_returns, cvar_tau);
        return {mean, stddev, cvar, catastrophic_count};
    }

    template <typename EnvT>
    ExperimentMetrics evaluate_root_metrics(
        std::shared_ptr<const EnvT> env,
        std::shared_ptr<const mcts::MctsDNode> root,
        const OptimalDistributionSolution& root_solution)
    {
        auto context = env->sample_context_itfc(env->get_initial_state_itfc());
        std::shared_ptr<const mcts::Action> recommended_action = root->recommend_action_itfc(*context);
        const int recommended_action_id =
            std::static_pointer_cast<const mcts::IntAction>(recommended_action)->action;

        ExperimentMetrics metrics{
            std::numeric_limits<double>::quiet_NaN(),
            0.0
        };

        for (int optimal_action_id : root_solution.optimal_actions) {
            if (recommended_action_id == optimal_action_id) {
                metrics.optimal_action_hit = 1.0;
                break;
            }
        }

        const auto catso_root = std::dynamic_pointer_cast<const mcts::CatsoDNode>(root);
        if (catso_root != nullptr) {
            auto mutable_catso_root = std::const_pointer_cast<mcts::CatsoDNode>(catso_root);
            auto root_actions = env->get_valid_actions_itfc(env->get_initial_state_itfc());

            double max_root_action_cvar = std::numeric_limits<double>::lowest();
            bool found_action_cvar = false;
            for (const auto& action : *root_actions) {
                std::shared_ptr<mcts::CatsoCNode> child;
                if (mutable_catso_root->has_child_node(action)) {
                    child = mutable_catso_root->get_child_node(action);
                }
                else {
                    child = mutable_catso_root->create_child_node(action);
                }

                max_root_action_cvar = std::max(max_root_action_cvar, child->get_cvar_value());
                found_action_cvar = true;
            }

            if (found_action_cvar) {
                metrics.cvar_regret = root_solution.optimal_cvar - max_root_action_cvar;
            }
        }

        return metrics;
    }

    inline std::vector<Candidate> build_candidates(
        double cvar_tau,
        int catso_n_atoms = 100,
        double optimism = 4.0,
        double power_mean_exponent = 1.0,
        int patso_particles = 100,
        double patso_optimism = 4.0,
        double uct_epsilon = 0.1)
    {
        std::vector<Candidate> cands;

        cands.push_back({"UCT", "bias=auto,epsilon=" + std::to_string(uct_epsilon), [=](auto env, auto init_state, int max_depth, int seed) {
            mcts::UctManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.bias = mcts::UctManagerArgs::USE_AUTO_BIAS;
            args.epsilon_exploration = uct_epsilon;
            args.seed = seed;
            auto mgr = std::make_shared<mcts::UctManager>(args);
            auto root = std::make_shared<mcts::UctDNode>(mgr, init_state, 0, 0);
            return std::make_pair(
                std::static_pointer_cast<mcts::MctsManager>(mgr),
                std::static_pointer_cast<mcts::MctsDNode>(root));
        }});

        cands.push_back({"CATSO", "n_atoms=" + std::to_string(catso_n_atoms)
                + ",optimism=" + std::to_string(optimism)
                + ",p=" + std::to_string(power_mean_exponent)
                + ",tau=" + std::to_string(cvar_tau),
            [=](auto env, auto init_state, int max_depth, int seed) {
                mcts::CatsoManagerArgs args(env);
                args.max_depth = max_depth;
                args.mcts_mode = false;
                args.n_atoms = catso_n_atoms;
                args.optimism_constant = optimism;
                args.power_mean_exponent = power_mean_exponent;
                args.cvar_tau = cvar_tau;
                args.seed = seed;
                auto mgr = std::make_shared<mcts::CatsoManager>(args);
                auto root = std::make_shared<mcts::CatsoDNode>(mgr, init_state, 0, 0);
                return std::make_pair(
                    std::static_pointer_cast<mcts::MctsManager>(mgr),
                    std::static_pointer_cast<mcts::MctsDNode>(root));
            }});

        cands.push_back({"PATSO", "max_particles=" + std::to_string(patso_particles)
                + ",optimism=" + std::to_string(patso_optimism)
                + ",p=" + std::to_string(power_mean_exponent)
                + ",tau=" + std::to_string(cvar_tau),
            [=](auto env, auto init_state, int max_depth, int seed) {
                mcts::PatsoManagerArgs args(env);
                args.max_depth = max_depth;
                args.mcts_mode = false;
                args.max_particles = patso_particles;
                args.optimism_constant = patso_optimism;
                args.power_mean_exponent = power_mean_exponent;
                args.cvar_tau = cvar_tau;
                args.seed = seed;
                auto mgr = std::make_shared<mcts::PatsoManager>(args);
                auto root = std::make_shared<mcts::PatsoDNode>(mgr, init_state, 0, 0);
                return std::make_pair(
                    std::static_pointer_cast<mcts::MctsManager>(mgr),
                    std::static_pointer_cast<mcts::MctsDNode>(root));
            }});

        return cands;
    }

    template <typename EnvT>
    SimpleEvalStats evaluate_cvar_only(
        std::shared_ptr<const EnvT> env,
        std::shared_ptr<const mcts::MctsDNode> root,
        int horizon,
        int rollouts,
        int threads,
        int seed,
        double cvar_tau,
        bool debug_trajectories = false,
        const std::string& debug_trajectory_label = "")
    {
        mcts::RandManager eval_rng(seed);
        mcts::EvalPolicy policy(root, env, eval_rng);
        mcts::MCEvaluator evaluator(
            env,
            policy,
            horizon,
            eval_rng,
            debug_trajectories,
            debug_trajectory_label);
        evaluator.run_rollouts(rollouts, threads);
        return {evaluator.get_cvar_return(cvar_tau), evaluator.get_stddev_cvar(cvar_tau)};
    }

    template <typename EnvT, typename StateT, typename CatastropheFn>
    int run_experiment(
        const std::string& env_id,
        const std::string& display_name,
        const std::string& extra_info,
        std::shared_ptr<EnvT> env,
        int horizon,
        double cvar_tau,
        int eval_rollouts,
        int runs,
        int threads,
        int base_seed,
        const std::vector<int>& trial_counts,
        const std::string& results_csv,
        const std::string& summary_csv,
        CatastropheFn catastrophe_fn,
        int catso_n_atoms = 100,
        double catso_optimism = 4.0,
        double power_mean_exponent = 1.0,
        int patso_particles = 100,
        double patso_optimism = 4.0,
        double uct_epsilon = 0.1)
    {
        const std::shared_ptr<const EnvT> const_env = env;
        std::shared_ptr<const StateT> typed_init_state = env->get_initial_state();
        std::shared_ptr<const mcts::State> init_state = env->get_initial_state_itfc();
        auto candidates = build_candidates(
            cvar_tau,
            catso_n_atoms,
            catso_optimism,
            power_mean_exponent,
            patso_particles,
            patso_optimism,
            uct_epsilon);
        CvarOracle<EnvT, StateT> oracle(const_env, cvar_tau);
        const auto& root_solution = oracle.solve_state(typed_init_state);
        std::map<std::pair<std::string, int>, SummaryAccumulator> summary_by_algo_and_trial;

        std::ofstream out(results_csv, std::ios::out | std::ios::trunc);
        out << std::setprecision(17);
        out << "env,algorithm,run,trial,mc_mean,mc_stddev,mc_cvar,cvar_regret,optimal_action_hit,catastrophic_count\n";

        std::cout << "[exp] " << display_name;
        if (!extra_info.empty()) {
            std::cout << ", " << extra_info;
        }
        std::cout << ", cvar_tau=" << cvar_tau << ", horizon=" << horizon << "\n";
        std::cout << "  Algorithms: " << candidates.size()
                  << ", Runs: " << runs
                  << ", Trial counts: " << trial_counts.size()
                  << "\n";
        std::cout << "  Root optimal CVaR: " << root_solution.optimal_cvar << "\n";
        const bool debug_eval_trajectories = mcts::should_debug_mc_eval_trajectories_from_env();

        for (const auto& cand : candidates) {
            bool printed_config = false;

            for (int run = 0; run < runs; ++run) {
                const int seed = base_seed + 1000 * run
                    + static_cast<int>(std::hash<std::string>{}(cand.algo) % 997);

                auto [mgr, root] = cand.make(
                    std::static_pointer_cast<mcts::MctsEnv>(env),
                    init_state,
                    horizon,
                    seed);
                if (!printed_config) {
                    const std::string runtime_config =
                        mcts::exp::describe_manager_config(std::static_pointer_cast<const mcts::MctsManager>(mgr));
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
                auto pool = std::make_unique<mcts::MctsPool>(mgr, root, threads);

                int trials_so_far = 0;
                for (int target_trials : trial_counts) {
                    const int trials_to_add = target_trials - trials_so_far;
                    if (trials_to_add > 0) {
                        pool->run_trials(trials_to_add, std::numeric_limits<double>::max(), true);
                        trials_so_far = target_trials;
                    }

                    const auto stats =
                        evaluate_tree(
                            const_env,
                            root,
                            horizon,
                            eval_rollouts,
                            threads,
                            seed + 777,
                            cvar_tau,
                            catastrophe_fn,
                            debug_eval_trajectories,
                            cand.algo);
                    const auto metrics = evaluate_root_metrics(const_env, root, root_solution);
                    out << env_id << "," << cand.algo << "," << run
                        << "," << target_trials << "," << stats.mean << "," << stats.stddev
                        << "," << stats.cvar
                        << "," << metrics.cvar_regret << "," << metrics.optimal_action_hit
                        << "," << stats.catastrophic_count << "\n";

                    SummaryAccumulator& summary = summary_by_algo_and_trial[{cand.algo, target_trials}];
                    summary.count++;
                    summary.sum_mc_mean += stats.mean;
                    summary.sum_mc_stddev += stats.stddev;
                    summary.sum_mc_cvar += stats.cvar;
                    summary.sum_optimal_action_hit += metrics.optimal_action_hit;
                    summary.sum_catastrophic_count += static_cast<double>(stats.catastrophic_count);
                    if (!std::isnan(metrics.cvar_regret)) {
                        summary.regret_count++;
                        summary.sum_cvar_regret += metrics.cvar_regret;
                    }
                }

                std::cout << "  " << cand.algo << " run " << (run + 1) << "/" << runs << " done\n";
            }
        }

        std::ofstream summary_out(summary_csv, std::ios::out | std::ios::trunc);
        summary_out << std::setprecision(17);
        summary_out << "env,algorithm,trial,mc_mean,mc_stddev,mc_cvar,cvar_regret,optimal_action_prob,catastrophic_count\n";
        for (const auto& [algo_trial, summary] : summary_by_algo_and_trial) {
            const std::string& algo = algo_trial.first;
            const int trial = algo_trial.second;
            const double mean_cvar_regret = (summary.regret_count > 0)
                ? (summary.sum_cvar_regret / static_cast<double>(summary.regret_count))
                : std::numeric_limits<double>::quiet_NaN();

            summary_out << env_id << "," << algo << "," << trial
                << "," << (summary.sum_mc_mean / static_cast<double>(summary.count))
                << "," << (summary.sum_mc_stddev / static_cast<double>(summary.count))
                << "," << (summary.sum_mc_cvar / static_cast<double>(summary.count))
                << "," << mean_cvar_regret
                << "," << (summary.sum_optimal_action_hit / static_cast<double>(summary.count))
                << "," << (summary.sum_catastrophic_count / static_cast<double>(summary.count))
                << "\n";
        }

        std::cout << "\nResults written to " << results_csv << " and " << summary_csv << "\n";
        return 0;
    }

    template <typename EnvT>
    int run_simple_experiment(
        const std::string& env_id,
        const std::string& display_name,
        const std::string& extra_info,
        std::shared_ptr<EnvT> env,
        int horizon,
        double cvar_tau,
        int eval_rollouts,
        int runs,
        int threads,
        int base_seed,
        const std::vector<int>& trial_counts,
        const std::string& results_csv,
        const std::string& summary_csv,
        int catso_n_atoms = 100,
        double catso_optimism = 4.0,
        double power_mean_exponent = 1.0,
        int patso_particles = 100,
        double patso_optimism = 4.0,
        double uct_epsilon = 0.1)
    {
        const std::shared_ptr<const EnvT> const_env = env;
        std::shared_ptr<const mcts::State> init_state = env->get_initial_state_itfc();
        auto candidates = build_candidates(
            cvar_tau,
            catso_n_atoms,
            catso_optimism,
            power_mean_exponent,
            patso_particles,
            patso_optimism,
            uct_epsilon);
        std::map<std::pair<std::string, int>, SimpleSummaryAccumulator> summary_by_algo_and_trial;

        std::ofstream out(results_csv, std::ios::out | std::ios::trunc);
        out << std::setprecision(17);
        out << "env,algorithm,run,trial,cvar_return,cvar_stddev\n";

        std::cout << "[exp] " << display_name;
        if (!extra_info.empty()) {
            std::cout << ", " << extra_info;
        }
        std::cout << ", cvar_tau=" << cvar_tau << ", horizon=" << horizon << "\n";
        std::cout << "  Algorithms: " << candidates.size()
                  << ", Runs: " << runs
                  << ", Trial counts: " << trial_counts.size()
                  << "\n";
        const bool debug_eval_trajectories = mcts::should_debug_mc_eval_trajectories_from_env();

        for (const auto& cand : candidates) {
            bool printed_config = false;

            for (int run = 0; run < runs; ++run) {
                const int seed = base_seed + 1000 * run
                    + static_cast<int>(std::hash<std::string>{}(cand.algo) % 997);

                auto [mgr, root] = cand.make(
                    std::static_pointer_cast<mcts::MctsEnv>(env),
                    init_state,
                    horizon,
                    seed);
                if (!printed_config) {
                    const std::string runtime_config =
                        mcts::exp::describe_manager_config(std::static_pointer_cast<const mcts::MctsManager>(mgr));
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
                auto pool = std::make_unique<mcts::MctsPool>(mgr, root, threads);

                int trials_so_far = 0;
                for (int target_trials : trial_counts) {
                    const int trials_to_add = target_trials - trials_so_far;
                    if (trials_to_add > 0) {
                        pool->run_trials(trials_to_add, std::numeric_limits<double>::max(), true);
                        trials_so_far = target_trials;
                    }

                    const auto stats = evaluate_cvar_only(
                        const_env,
                        root,
                        horizon,
                        eval_rollouts,
                        threads,
                        seed + 777,
                        cvar_tau,
                        debug_eval_trajectories,
                        cand.algo);
                    out << env_id << "," << cand.algo << "," << run
                        << "," << target_trials
                        << "," << stats.cvar_return
                        << "," << stats.cvar_stddev
                        << "\n";

                    SimpleSummaryAccumulator& summary = summary_by_algo_and_trial[{cand.algo, target_trials}];
                    summary.count++;
                    summary.sum_cvar_return += stats.cvar_return;
                    summary.sum_cvar_stddev += stats.cvar_stddev;
                }

                std::cout << "  " << cand.algo << " run " << (run + 1) << "/" << runs << " done\n";
            }
        }

        std::ofstream summary_out(summary_csv, std::ios::out | std::ios::trunc);
        summary_out << std::setprecision(17);
        summary_out << "env,algorithm,trial,cvar_return,cvar_stddev\n";
        for (const auto& [algo_trial, summary] : summary_by_algo_and_trial) {
            const std::string& algo = algo_trial.first;
            const int trial = algo_trial.second;
            summary_out << env_id << "," << algo << "," << trial
                << "," << (summary.sum_cvar_return / static_cast<double>(summary.count))
                << "," << (summary.sum_cvar_stddev / static_cast<double>(summary.count))
                << "\n";
        }

        std::cout << "\nResults written to " << results_csv << " and " << summary_csv << "\n";
        return 0;
    }
}
