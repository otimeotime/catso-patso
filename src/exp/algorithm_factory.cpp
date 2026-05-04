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
#include <stdexcept>
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
        double discount_gamma,
        const RunCandidateConfig& config)
    {
        std::vector<Candidate> candidates;

        if (config.uct.enabled) {
            candidates.push_back({
                "UCT",
                std::string("bias=")
                    + (config.uct.use_auto_bias ? "auto" : format_double(config.uct.bias))
                    + ",epsilon=" + format_double(config.uct.epsilon),
                cvar_tau,
                [=](auto env, auto init_state, int max_depth, int seed) {
                    mcts::UctManagerArgs args(env);
                    args.max_depth = max_depth;
                    args.mcts_mode = false;
                    args.bias = config.uct.use_auto_bias ? mcts::UctManagerArgs::USE_AUTO_BIAS : config.uct.bias;
                    args.epsilon_exploration = config.uct.epsilon;
                    args.seed = seed;

                    auto mgr = std::make_shared<mcts::UctManager>(args);
                    auto root = std::make_shared<mcts::UctDNode>(mgr, init_state, 0, 0);
                    return std::make_pair(
                        std::static_pointer_cast<mcts::MctsManager>(mgr),
                        std::static_pointer_cast<mcts::MctsDNode>(root));
                }
            });
        }

        if (config.max_uct.enabled) {
            candidates.push_back({
                "MaxUCT",
                std::string("bias=")
                    + (config.max_uct.use_auto_bias ? "auto" : format_double(config.max_uct.bias))
                    + ",epsilon=" + format_double(config.max_uct.epsilon),
                cvar_tau,
                [=](auto env, auto init_state, int max_depth, int seed) {
                    mcts::UctManagerArgs args(env);
                    args.max_depth = max_depth;
                    args.mcts_mode = false;
                    args.bias =
                        config.max_uct.use_auto_bias
                        ? mcts::UctManagerArgs::USE_AUTO_BIAS
                        : config.max_uct.bias;
                    args.recommend_most_visited = false;
                    args.epsilon_exploration = config.max_uct.epsilon;
                    args.seed = seed;

                    auto mgr = std::make_shared<mcts::UctManager>(args);
                    auto root = std::make_shared<mcts::MaxUctDNode>(mgr, init_state, 0, 0);
                    return std::make_pair(
                        std::static_pointer_cast<mcts::MctsManager>(mgr),
                        std::static_pointer_cast<mcts::MctsDNode>(root));
                }
            });
        }

        if (config.power_uct.enabled) {
            candidates.push_back({
                "PowerUCT",
                std::string("bias=")
                    + (config.power_uct.use_auto_bias ? "auto" : format_double(config.power_uct.bias))
                    + ",epsilon=" + format_double(config.power_uct.epsilon)
                    + ",p=" + format_double(config.power_uct.power_mean_exponent),
                cvar_tau,
                [=](auto env, auto init_state, int max_depth, int seed) {
                    mcts::PowerUctManagerArgs args(env, config.power_uct.power_mean_exponent);
                    args.max_depth = max_depth;
                    args.mcts_mode = false;
                    args.bias =
                        config.power_uct.use_auto_bias
                        ? mcts::UctManagerArgs::USE_AUTO_BIAS
                        : config.power_uct.bias;
                    args.recommend_most_visited = false;
                    args.epsilon_exploration = config.power_uct.epsilon;
                    args.seed = seed;

                    auto mgr = std::make_shared<mcts::PowerUctManager>(args);
                    auto root = std::make_shared<mcts::PowerUctDNode>(mgr, init_state, 0, 0);
                    return std::make_pair(
                        std::static_pointer_cast<mcts::MctsManager>(mgr),
                        std::static_pointer_cast<mcts::MctsDNode>(root));
                }
            });
        }

        if (config.catso.enabled) {
            candidates.push_back({
                "CATSO",
                "n_atoms=" + std::to_string(config.catso.n_atoms)
                    + ",optimism=" + format_double(config.catso.optimism)
                    + ",p=" + format_double(config.catso.power_mean_exponent)
                    + ",cvar_tau=" + format_double(cvar_tau)
                    + ",gamma=" + format_double(discount_gamma),
                cvar_tau,
                [=](auto env, auto init_state, int max_depth, int seed) {
                    mcts::CatsoManagerArgs args(env);
                    args.max_depth = max_depth;
                    args.mcts_mode = false;
                    args.n_atoms = config.catso.n_atoms;
                    args.optimism_constant = config.catso.optimism;
                    args.power_mean_exponent = config.catso.power_mean_exponent;
                    args.cvar_tau = cvar_tau;
                    args.discount_gamma = discount_gamma;
                    args.seed = seed;

                    auto mgr = std::make_shared<mcts::CatsoManager>(args);
                    auto root = std::make_shared<mcts::CatsoDNode>(mgr, init_state, 0, 0);
                    return std::make_pair(
                        std::static_pointer_cast<mcts::MctsManager>(mgr),
                        std::static_pointer_cast<mcts::MctsDNode>(root));
                }
            });
        }

        if (config.patso.enabled) {
            candidates.push_back({
                "PATSO",
                "max_particles=" + std::to_string(config.patso.max_particles)
                    + ",optimism=" + format_double(config.patso.optimism)
                    + ",p=" + format_double(config.patso.power_mean_exponent)
                    + ",cvar_tau=" + format_double(cvar_tau)
                    + ",gamma=" + format_double(discount_gamma),
                cvar_tau,
                [=](auto env, auto init_state, int max_depth, int seed) {
                    mcts::PatsoManagerArgs args(env);
                    args.max_depth = max_depth;
                    args.mcts_mode = false;
                    args.max_particles = config.patso.max_particles;
                    args.optimism_constant = config.patso.optimism;
                    args.power_mean_exponent = config.patso.power_mean_exponent;
                    args.cvar_tau = cvar_tau;
                    args.discount_gamma = discount_gamma;
                    args.seed = seed;

                    auto mgr = std::make_shared<mcts::PatsoManager>(args);
                    auto root = std::make_shared<mcts::PatsoDNode>(mgr, init_state, 0, 0);
                    return std::make_pair(
                        std::static_pointer_cast<mcts::MctsManager>(mgr),
                        std::static_pointer_cast<mcts::MctsDNode>(root));
                }
            });
        }

        if (candidates.empty()) {
            throw std::runtime_error("No algorithms enabled in RunCandidateConfig");
        }

        return candidates;
    }

    std::vector<Candidate> build_default_tuning_candidates(
        double default_eval_tau,
        double discount_gamma,
        const TuningGridConfig& config)
    {
        std::vector<Candidate> candidates;

        if (config.uct.enabled) {
            for (double bias : config.uct.bias_values) {
                for (double epsilon : config.uct.epsilon_values) {
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
                }
            }
        }

        if (config.max_uct.enabled) {
            for (double bias : config.max_uct.bias_values) {
                for (double epsilon : config.max_uct.epsilon_values) {
                    const std::string bias_str =
                        (bias == mcts::UctManagerArgs::USE_AUTO_BIAS) ? "auto" : format_double(bias);
                    candidates.push_back({
                        "MaxUCT",
                        "bias=" + bias_str + ",eps=" + format_double(epsilon),
                        default_eval_tau,
                        [=](auto env, auto init_state, int max_depth, int seed) {
                            mcts::UctManagerArgs args(env);
                            args.max_depth = max_depth;
                            args.mcts_mode = false;
                            args.bias = bias;
                            args.recommend_most_visited = false;
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
        }

        if (config.power_uct.enabled) {
            for (double p_mean : config.power_uct.power_mean_exponent_values) {
                for (double bias : config.power_uct.bias_values) {
                    for (double epsilon : config.power_uct.epsilon_values) {
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
                                args.recommend_most_visited = false;
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
        }

        if (config.catso.enabled) {
            for (double p_mean : config.catso.power_mean_exponent_values) {
                for (int n_atoms : config.catso.n_atoms_values) {
                    for (double optimism : config.catso.optimism_values) {
                        for (double tau : config.catso.tau_values) {
                            candidates.push_back({
                                "CATSO",
                                "n_atoms=" + std::to_string(n_atoms)
                                    + ",optimism=" + format_double(optimism)
                                    + ",p=" + format_double(p_mean)
                                    + ",cvar_tau=" + format_double(tau)
                                    + ",gamma=" + format_double(discount_gamma),
                                tau,
                                [=](auto env, auto init_state, int max_depth, int seed) {
                                    mcts::CatsoManagerArgs args(env);
                                    args.max_depth = max_depth;
                                    args.mcts_mode = false;
                                    args.n_atoms = n_atoms;
                                    args.optimism_constant = optimism;
                                    args.power_mean_exponent = p_mean;
                                    args.cvar_tau = tau;
                                    args.discount_gamma = discount_gamma;
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
        }

        if (config.patso.enabled) {
            for (double p_mean : config.patso.power_mean_exponent_values) {
                for (int max_particles : config.patso.max_particles_values) {
                    for (double optimism : config.patso.optimism_values) {
                        for (double tau : config.patso.tau_values) {
                            candidates.push_back({
                                "PATSO",
                                "max_particles=" + std::to_string(max_particles)
                                    + ",optimism=" + format_double(optimism)
                                    + ",p=" + format_double(p_mean)
                                    + ",cvar_tau=" + format_double(tau)
                                    + ",gamma=" + format_double(discount_gamma),
                                tau,
                                [=](auto env, auto init_state, int max_depth, int seed) {
                                    mcts::PatsoManagerArgs args(env);
                                    args.max_depth = max_depth;
                                    args.mcts_mode = false;
                                    args.max_particles = max_particles;
                                    args.optimism_constant = optimism;
                                    args.power_mean_exponent = p_mean;
                                    args.cvar_tau = tau;
                                    args.discount_gamma = discount_gamma;
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
        }

        if (candidates.empty()) {
            throw std::runtime_error("No algorithms enabled in TuningGridConfig");
        }

        return candidates;
    }
}
