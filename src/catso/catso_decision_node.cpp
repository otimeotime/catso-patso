#include "algorithms/catso/catso_decision_node.h"

#include "algorithms/catso/catso_chance_node.h"
#include "algorithms/catso/catso_manager.h"
#include "helper_templates.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace {
    constexpr double kPowerMeanShiftEps = 1e-9;
}

namespace mcts {
    CatsoDNode::CatsoDNode(
        shared_ptr<CatsoManager> mcts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const CatsoCNode> parent) :
            MctsDNode(
                static_pointer_cast<MctsManager>(mcts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MctsCNode>(parent)),
            value_estimate(heuristic_value),
            actions(mcts_manager->mcts_env->get_valid_actions_itfc(state))
    {
    }

    void CatsoDNode::visit(MctsEnvContext& ctx) {
        MctsDNode::visit_itfc(ctx);
    }

    double CatsoDNode::compute_optimism_bonus(int parent_visits, int child_visits) const {
        if (child_visits <= 0) {
            return numeric_limits<double>::infinity();
        }

        const double parent_visits_d = static_cast<double>(max(parent_visits, 1));
        const double child_visits_d = static_cast<double>(child_visits);
        CatsoManager& manager = (CatsoManager&) *mcts_manager;
        return manager.optimism_constant * pow(parent_visits_d, 0.25) / sqrt(child_visits_d);
    }

    double CatsoDNode::shifted_weighted_power_mean(
        const vector<pair<double,double>>& weighted_values,
        double exponent) const
    {
        if (weighted_values.empty()) {
            return 0.0;
        }

        double sum_weights = 0.0;
        double min_value = numeric_limits<double>::infinity();
        double weighted_mean = 0.0;
        for (const auto& pr : weighted_values) {
            if (pr.first <= 0.0) {
                continue;
            }
            sum_weights += pr.first;
            weighted_mean += pr.first * pr.second;
            min_value = min(min_value, pr.second);
        }

        if (sum_weights <= 0.0) {
            return 0.0;
        }

        if (abs(exponent - 1.0) < 1e-12) {
            return weighted_mean / sum_weights;
        }

        const double offset = (min_value <= 0.0) ? (-min_value + kPowerMeanShiftEps) : 0.0;
        double powered_sum = 0.0;
        for (const auto& pr : weighted_values) {
            if (pr.first <= 0.0) {
                continue;
            }
            const double shifted_val = max(pr.second + offset, 0.0);
            powered_sum += (pr.first / sum_weights) * pow(shifted_val, exponent);
        }

        return pow(max(powered_sum, 0.0), 1.0 / exponent) - offset;
    }

    double CatsoDNode::compute_power_mean_value() const {
        const double opp_coeff = is_opponent() ? -1.0 : 1.0;
        vector<pair<double,double>> weighted_values;

        lock_all_children();
        for (shared_ptr<const Action> action : *actions) {
            if (!has_child_node(action)) {
                continue;
            }

            shared_ptr<CatsoCNode> child = get_child_node(action);
            if (child->num_visits <= 0) {
                continue;
            }

            weighted_values.emplace_back(
                static_cast<double>(child->num_visits),
                opp_coeff * child->get_mean_value());
        }
        unlock_all_children();

        CatsoManager& manager = (CatsoManager&) *mcts_manager;
        return opp_coeff * shifted_weighted_power_mean(weighted_values, manager.power_mean_exponent);
    }

    void CatsoDNode::fill_ts_values(
        unordered_map<shared_ptr<const Action>,double>& action_values,
        MctsEnvContext& ctx) const
    {
        (void)ctx;
        const double opp_coeff = is_opponent() ? -1.0 : 1.0;

        lock_all_children();
        for (shared_ptr<const Action> action : *actions) {
            if (!has_child_node(action)) {
                continue;
            }

            shared_ptr<CatsoCNode> child = get_child_node(action);
            double score = opp_coeff * child->sample_cvar_value();
            score += compute_optimism_bonus(num_visits, child->num_visits);
            action_values[action] = score;
        }
        unlock_all_children();
    }

    shared_ptr<const Action> CatsoDNode::select_action_ts(MctsEnvContext& ctx) {
        vector<shared_ptr<const Action>> unexpanded_actions;
        for (shared_ptr<const Action> action : *actions) {
            if (!has_child_node(action)) {
                unexpanded_actions.push_back(action);
            }
        }

        if (!unexpanded_actions.empty()) {
            const int idx = mcts_manager->get_rand_int(0, static_cast<int>(unexpanded_actions.size()));
            shared_ptr<const Action> action = unexpanded_actions[static_cast<size_t>(idx)];
            create_child_node(action);
            return action;
        }

        unordered_map<shared_ptr<const Action>,double> action_values;
        fill_ts_values(action_values, ctx);
        shared_ptr<const Action> result_action =
            helper::get_max_key_break_ties_randomly(action_values, *mcts_manager);

        if (!has_child_node(result_action)) {
            create_child_node(result_action);
        }
        return result_action;
    }

    shared_ptr<const Action> CatsoDNode::select_action(MctsEnvContext& ctx) {
        return select_action_ts(ctx);
    }

    shared_ptr<const Action> CatsoDNode::recommend_action_best_cvar() const {
        const double opp_coeff = is_opponent() ? -1.0 : 1.0;
        unordered_map<shared_ptr<const Action>,double> action_values;

        for (shared_ptr<const Action> action : *actions) {
            if (!has_child_node(action)) {
                continue;
            }
            action_values[action] = opp_coeff * get_child_node(action)->get_cvar_value();
        }

        if (action_values.empty()) {
            const int idx = mcts_manager->get_rand_int(0, static_cast<int>(actions->size()));
            return actions->at(static_cast<size_t>(idx));
        }

        return helper::get_max_key_break_ties_randomly(action_values, *mcts_manager);
    }

    shared_ptr<const Action> CatsoDNode::recommend_action(MctsEnvContext& ctx) const {
        (void)ctx;
        return recommend_action_best_cvar();
    }

    void CatsoDNode::backup(
        const vector<double>& trial_rewards_before_node,
        const vector<double>& trial_rewards_after_node,
        const double trial_cumulative_return_after_node,
        const double trial_cumulative_return,
        MctsEnvContext& ctx)
    {
        (void)trial_rewards_before_node;
        (void)trial_rewards_after_node;
        (void)trial_cumulative_return_after_node;
        (void)trial_cumulative_return;
        (void)ctx;

        value_estimate = compute_power_mean_value();
    }

    shared_ptr<CatsoCNode> CatsoDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<CatsoCNode>(
            static_pointer_cast<CatsoManager>(mcts_manager),
            state,
            action,
            decision_depth,
            decision_timestep,
            static_pointer_cast<const CatsoDNode>(shared_from_this()));
    }

    string CatsoDNode::get_pretty_print_val() const {
        stringstream ss;
        ss << value_estimate;
        return ss.str();
    }
}

namespace mcts {
    shared_ptr<CatsoCNode> CatsoDNode::create_child_node(shared_ptr<const Action> action) {
        shared_ptr<MctsCNode> new_child = MctsDNode::create_child_node_itfc(action);
        return static_pointer_cast<CatsoCNode>(new_child);
    }

    bool CatsoDNode::has_child_node(shared_ptr<const Action> action) const {
        return MctsDNode::has_child_node_itfc(static_pointer_cast<const Action>(action));
    }

    shared_ptr<CatsoCNode> CatsoDNode::get_child_node(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<MctsCNode> new_child = MctsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<CatsoCNode>(new_child);
    }
}

namespace mcts {
    void CatsoDNode::visit_itfc(MctsEnvContext& ctx) {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> CatsoDNode::select_action_itfc(MctsEnvContext& ctx) {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        return select_action(ctx_itfc);
    }

    shared_ptr<const Action> CatsoDNode::recommend_action_itfc(MctsEnvContext& ctx) const {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        return recommend_action(ctx_itfc);
    }

    void CatsoDNode::backup_itfc(
        const vector<double>& trial_rewards_before_node,
        const vector<double>& trial_rewards_after_node,
        const double trial_cumulative_return_after_node,
        const double trial_cumulative_return,
        MctsEnvContext& ctx)
    {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        backup(
            trial_rewards_before_node,
            trial_rewards_after_node,
            trial_cumulative_return_after_node,
            trial_cumulative_return,
            ctx_itfc);
    }

    shared_ptr<MctsCNode> CatsoDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<CatsoCNode> child_node = create_child_node_helper(action);
        return static_pointer_cast<MctsCNode>(child_node);
    }
}
