#include "exp/algorithm_factory.h"

#include "algorithms/catso/catso_decision_node.h"
#include "algorithms/catso/catso_manager.h"
#include "algorithms/catso/patso_decision_node.h"
#include "algorithms/catso/patso_manager.h"
#include "algorithms/uct/max_uct_decision_node.h"
#include "algorithms/uct/power_uct_decision_node.h"
#include "algorithms/uct/power_uct_manager.h"
#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/uct_manager.h"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mcts::exp {
    namespace {
        std::string format_double(double value) {
            std::ostringstream ss;
            ss << value;
            return ss.str();
        }
    }

    std::vector<Candidate> build_default_run_candidates(
        double cvar_tau,
        const RunCandidateConfig& config)
    {
        std::vector<Candidate> candidates;

        candidates.push_back({
            "UCT",
            std::string("bias=")
                + (config.uct_use_auto_bias ? "auto" : format_double(config.uct_bias))
                + ",epsilon=" + format_double(config.uct_epsilon),
            cvar_tau,
            [=](auto env, auto init_state, int max_depth, int seed) {
                mcts::UctManagerArgs args(env);
                args.max_depth = max_depth;
                args.mcts_mode = false;
                args.bias = config.uct_use_auto_bias ? mcts::UctManagerArgs::USE_AUTO_BIAS : config.uct_bias;
                args.epsilon_exploration = config.uct_epsilon;
                args.seed = seed;

                auto mgr = std::make_shared<mcts::UctManager>(args);
                auto root = std::make_shared<mcts::UctDNode>(mgr, init_state, 0, 0);
                return std::make_pair(
                    std::static_pointer_cast<mcts::MctsManager>(mgr),
                    std::static_pointer_cast<mcts::MctsDNode>(root));
            }
        });

        candidates.push_back({
            "MaxUCT",
            std::string("bias=")
                + (config.uct_use_auto_bias ? "auto" : format_double(config.uct_bias))
                + ",epsilon=" + format_double(config.uct_epsilon),
            cvar_tau,
            [=](auto env, auto init_state, int max_depth, int seed) {
                mcts::UctManagerArgs args(env);
                args.max_depth = max_depth;
                args.mcts_mode = false;
                args.bias = config.uct_use_auto_bias ? mcts::UctManagerArgs::USE_AUTO_BIAS : config.uct_bias;
                args.epsilon_exploration = config.uct_epsilon;
                args.seed = seed;

                auto mgr = std::make_shared<mcts::UctManager>(args);
                auto root = std::make_shared<mcts::MaxUctDNode>(mgr, init_state, 0, 0);
                return std::make_pair(
                    std::static_pointer_cast<mcts::MctsManager>(mgr),
                    std::static_pointer_cast<mcts::MctsDNode>(root));
            }
        });

        candidates.push_back({
            "PowerUCT",
            std::string("bias=")
                + (config.uct_use_auto_bias ? "auto" : format_double(config.uct_bias))
                + ",epsilon=" + format_double(config.uct_epsilon)
                + ",p=" + format_double(config.power_mean_exponent),
            cvar_tau,
            [=](auto env, auto init_state, int max_depth, int seed) {
                mcts::PowerUctManagerArgs args(env, config.power_mean_exponent);
                args.max_depth = max_depth;
                args.mcts_mode = false;
                args.bias = config.uct_use_auto_bias ? mcts::UctManagerArgs::USE_AUTO_BIAS : config.uct_bias;
                args.epsilon_exploration = config.uct_epsilon;
                args.seed = seed;

                auto mgr = std::make_shared<mcts::PowerUctManager>(args);
                auto root = std::make_shared<mcts::PowerUctDNode>(mgr, init_state, 0, 0);
                return std::make_pair(
                    std::static_pointer_cast<mcts::MctsManager>(mgr),
                    std::static_pointer_cast<mcts::MctsDNode>(root));
            }
        });

        candidates.push_back({
            "CATSO",
            "n_atoms=" + std::to_string(config.catso_n_atoms)
                + ",optimism=" + format_double(config.catso_optimism)
                + ",p=" + format_double(config.power_mean_exponent)
                + ",cvar_tau=" + format_double(cvar_tau),
            cvar_tau,
            [=](auto env, auto init_state, int max_depth, int seed) {
                mcts::CatsoManagerArgs args(env);
                args.max_depth = max_depth;
                args.mcts_mode = false;
                args.n_atoms = config.catso_n_atoms;
                args.optimism_constant = config.catso_optimism;
                args.power_mean_exponent = config.power_mean_exponent;
                args.cvar_tau = cvar_tau;
                args.seed = seed;

                auto mgr = std::make_shared<mcts::CatsoManager>(args);
                auto root = std::make_shared<mcts::CatsoDNode>(mgr, init_state, 0, 0);
                return std::make_pair(
                    std::static_pointer_cast<mcts::MctsManager>(mgr),
                    std::static_pointer_cast<mcts::MctsDNode>(root));
            }
        });

        candidates.push_back({
            "PATSO",
            "max_particles=" + std::to_string(config.patso_particles)
                + ",optimism=" + format_double(config.patso_optimism)
                + ",p=" + format_double(config.power_mean_exponent)
                + ",cvar_tau=" + format_double(cvar_tau),
            cvar_tau,
            [=](auto env, auto init_state, int max_depth, int seed) {
                mcts::PatsoManagerArgs args(env);
                args.max_depth = max_depth;
                args.mcts_mode = false;
                args.max_particles = config.patso_particles;
                args.optimism_constant = config.patso_optimism;
                args.power_mean_exponent = config.power_mean_exponent;
                args.cvar_tau = cvar_tau;
                args.seed = seed;

                auto mgr = std::make_shared<mcts::PatsoManager>(args);
                auto root = std::make_shared<mcts::PatsoDNode>(mgr, init_state, 0, 0);
                return std::make_pair(
                    std::static_pointer_cast<mcts::MctsManager>(mgr),
                    std::static_pointer_cast<mcts::MctsDNode>(root));
            }
        });

        return candidates;
    }

    std::vector<Candidate> build_default_tuning_candidates(
        double default_eval_tau,
        const TuningGridConfig& config)
    {
        std::vector<Candidate> candidates;

        for (double bias : config.uct_bias_values) {
            for (double epsilon : config.uct_epsilon_values) {
                const std::string bias_str =
                    (bias == mcts::UctManagerArgs::USE_AUTO_BIAS) ? "auto" : format_double(bias);
                candidates.push_back({
                    "UCT",
                    "bias=" + bias_str + ",eps=" + format_double(epsilon),
                    default_eval_tau,
                    [=](auto env, auto init_state, int max_depth, int seed) {
                        mcts::UctManagerArgs args(env);
                        args.max_depth = max_depth;
                        args.mcts_mode = false;
                        args.bias = bias;
                        args.epsilon_exploration = epsilon;
                        args.seed = seed;

                        auto mgr = std::make_shared<mcts::UctManager>(args);
                        auto root = std::make_shared<mcts::UctDNode>(mgr, init_state, 0, 0);
                        return std::make_pair(
                            std::static_pointer_cast<mcts::MctsManager>(mgr),
                            std::static_pointer_cast<mcts::MctsDNode>(root));
                    }
                });

                candidates.push_back({
                    "MaxUCT",
                    "bias=" + bias_str + ",eps=" + format_double(epsilon),
                    default_eval_tau,
                    [=](auto env, auto init_state, int max_depth, int seed) {
                        mcts::UctManagerArgs args(env);
                        args.max_depth = max_depth;
                        args.mcts_mode = false;
                        args.bias = bias;
                        args.epsilon_exploration = epsilon;
                        args.seed = seed;

                        auto mgr = std::make_shared<mcts::UctManager>(args);
                        auto root = std::make_shared<mcts::MaxUctDNode>(mgr, init_state, 0, 0);
                        return std::make_pair(
                            std::static_pointer_cast<mcts::MctsManager>(mgr),
                            std::static_pointer_cast<mcts::MctsDNode>(root));
                    }
                });
            }
        }

        for (double p_mean : config.power_mean_exponent_values) {
            for (double bias : config.uct_bias_values) {
                for (double epsilon : config.uct_epsilon_values) {
                    const std::string bias_str =
                        (bias == mcts::UctManagerArgs::USE_AUTO_BIAS) ? "auto" : format_double(bias);
                    candidates.push_back({
                        "PowerUCT",
                        "bias=" + bias_str
                            + ",eps=" + format_double(epsilon)
                            + ",p=" + format_double(p_mean),
                        default_eval_tau,
                        [=](auto env, auto init_state, int max_depth, int seed) {
                            mcts::PowerUctManagerArgs args(env, p_mean);
                            args.max_depth = max_depth;
                            args.mcts_mode = false;
                            args.bias = bias;
                            args.epsilon_exploration = epsilon;
                            args.seed = seed;

                            auto mgr = std::make_shared<mcts::PowerUctManager>(args);
                            auto root = std::make_shared<mcts::PowerUctDNode>(mgr, init_state, 0, 0);
                            return std::make_pair(
                                std::static_pointer_cast<mcts::MctsManager>(mgr),
                                std::static_pointer_cast<mcts::MctsDNode>(root));
                        }
                    });
                }
            }
        }

        for (double p_mean : {1.0}) {
            for (int n_atoms : config.catso_n_atoms_values) {
                for (double optimism : config.catso_optimism_values) {
                    for (double tau : config.tau_values) {
                        candidates.push_back({
                            "CATSO",
                            "n_atoms=" + std::to_string(n_atoms)
                                + ",optimism=" + format_double(optimism)
                                + ",p=" + format_double(p_mean)
                                + ",cvar_tau=" + format_double(tau),
                            tau,
                            [=](auto env, auto init_state, int max_depth, int seed) {
                                mcts::CatsoManagerArgs args(env);
                                args.max_depth = max_depth;
                                args.mcts_mode = false;
                                args.n_atoms = n_atoms;
                                args.optimism_constant = optimism;
                                args.power_mean_exponent = p_mean;
                                args.cvar_tau = tau;
                                args.seed = seed;

                                auto mgr = std::make_shared<mcts::CatsoManager>(args);
                                auto root = std::make_shared<mcts::CatsoDNode>(mgr, init_state, 0, 0);
                                return std::make_pair(
                                    std::static_pointer_cast<mcts::MctsManager>(mgr),
                                    std::static_pointer_cast<mcts::MctsDNode>(root));
                            }
                        });
                    }
                }
            }
        }

        for (double p_mean : {1.0}) {
            for (int max_particles : config.patso_particles_values) {
                for (double optimism : config.patso_optimism_values) {
                    for (double tau : config.tau_values) {
                        candidates.push_back({
                            "PATSO",
                            "max_particles=" + std::to_string(max_particles)
                                + ",optimism=" + format_double(optimism)
                                + ",p=" + format_double(p_mean)
                                + ",cvar_tau=" + format_double(tau),
                            tau,
                            [=](auto env, auto init_state, int max_depth, int seed) {
                                mcts::PatsoManagerArgs args(env);
                                args.max_depth = max_depth;
                                args.mcts_mode = false;
                                args.max_particles = max_particles;
                                args.optimism_constant = optimism;
                                args.power_mean_exponent = p_mean;
                                args.cvar_tau = tau;
                                args.seed = seed;

                                auto mgr = std::make_shared<mcts::PatsoManager>(args);
                                auto root = std::make_shared<mcts::PatsoDNode>(mgr, init_state, 0, 0);
                                return std::make_pair(
                                    std::static_pointer_cast<mcts::MctsManager>(mgr),
                                    std::static_pointer_cast<mcts::MctsDNode>(root));
                            }
                        });
                    }
                }
            }
        }

        return candidates;
    }
}
