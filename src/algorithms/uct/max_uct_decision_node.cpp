#include "algorithms/uct/max_uct_decision_node.h"

#include "algorithms/uct/max_uct_chance_node.h"
#include "helper_templates.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

using namespace std;

namespace mcts {
    MaxUctDNode::MaxUctDNode(
        shared_ptr<UctManager> mcts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const MaxUctCNode> parent) :
            MctsDNode(
                static_pointer_cast<MctsManager>(mcts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MctsCNode>(parent)),
            num_backups(0),
            max_return(0.0),
            actions(mcts_manager->mcts_env->get_valid_actions_itfc(state)),
            policy_prior()
    {
        if (mcts_manager->heuristic_fn != nullptr) {
            num_visits = mcts_manager->heuristic_psuedo_trials;
            num_backups = mcts_manager->heuristic_psuedo_trials;
            max_return = heuristic_value;
        }

        if (mcts_manager->prior_fn != nullptr) {
            policy_prior = mcts_manager->prior_fn(state, mcts_manager->mcts_env);
        }
    }

    bool MaxUctDNode::has_prior() const {
        UctManager& manager = *static_pointer_cast<UctManager>(mcts_manager);
        return manager.prior_fn != nullptr;
    }

    void MaxUctDNode::visit(MctsEnvContext& ctx) {
        MctsDNode::visit_itfc(ctx);
    }

    double MaxUctDNode::compute_ucb_term(int num_visits, int child_visits) const {
        double num_visits_d = (num_visits > 0) ? static_cast<double>(num_visits) : 1.0;
        double child_visits_d = (child_visits > 0) ? static_cast<double>(child_visits) : 1.0;
        return sqrt(log(num_visits_d) / child_visits_d);
    }

    void MaxUctDNode::fill_ucb_values(
        unordered_map<shared_ptr<const Action>, double>& ucb_values,
        MctsEnvContext& ctx) const
    {
        shared_ptr<UctManager> manager = static_pointer_cast<UctManager>(mcts_manager);
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        (void) ctx;

        lock_all_children();

        double bias = manager->bias;
        if (bias == UctManager::USE_AUTO_BIAS) {
            bias = UctManager::AUTO_BIAS_MIN_BIAS;
            for (shared_ptr<const Action> action : *actions) {
                if (!has_child_node(action)) {
                    continue;
                }
                double child_abs_val = abs(get_child_node(action)->max_return);
                if (child_abs_val > bias) {
                    bias = child_abs_val;
                }
            }
        }

        for (shared_ptr<const Action> action : *actions) {
            double action_ucb_value = 0.0;

            int child_visits = has_child_node(action) ? get_child_node(action)->num_visits : 0;
            action_ucb_value += compute_ucb_term(num_visits, child_visits);
            action_ucb_value *= bias;
            if (has_prior()) {
                action_ucb_value *= policy_prior->at(action);
            }

            if (has_child_node(action)) {
                action_ucb_value += opp_coeff * get_child_node(action)->max_return;
            }

            ucb_values[action] = action_ucb_value;
        }

        unlock_all_children();
    }

    shared_ptr<const Action> MaxUctDNode::select_action_ucb(MctsEnvContext& ctx) {
        if (!has_prior()) {
            vector<shared_ptr<const Action>> actions_yet_to_try;
            for (shared_ptr<const Action> action : *actions) {
                if (!has_child_node(action)) {
                    actions_yet_to_try.push_back(action);
                }
            }

            if (!actions_yet_to_try.empty()) {
                int indx = mcts_manager->get_rand_int(0, actions_yet_to_try.size());
                shared_ptr<const Action> action = actions_yet_to_try[indx];
                create_child_node(action);
                return action;
            }
        }

        unordered_map<shared_ptr<const Action>, double> ucb_values;
        fill_ucb_values(ucb_values, ctx);
        shared_ptr<const Action> result_action = helper::get_max_key_break_ties_randomly(ucb_values, *mcts_manager);

        if (!has_child_node(result_action)) {
            create_child_node(result_action);
        }
        return result_action;
    }

    shared_ptr<const Action> MaxUctDNode::select_action_random() {
        int index = mcts_manager->get_rand_int(0, actions->size());
        shared_ptr<const Action> action = actions->at(index);
        if (!has_child_node(action)) {
            create_child_node(action);
        }
        return action;
    }

    shared_ptr<const Action> MaxUctDNode::select_action(MctsEnvContext& ctx) {
        shared_ptr<UctManager> manager = static_pointer_cast<UctManager>(mcts_manager);
        if (manager->epsilon_exploration > 0.0) {
            if (manager->get_rand_uniform() < manager->epsilon_exploration) {
                return select_action_random();
            }
        }
        return select_action_ucb(ctx);
    }

    shared_ptr<const Action> MaxUctDNode::recommend_action_best_empirical() const {
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        unordered_map<shared_ptr<const Action>, double> action_values;

        for (shared_ptr<const Action> action : *actions) {
            if (!has_child_node(action)) {
                continue;
            }
            action_values[action] = opp_coeff * get_child_node(action)->max_return;
        }

        if (action_values.empty()) {
            int index = mcts_manager->get_rand_int(0, actions->size());
            return actions->at(index);
        }

        return helper::get_max_key_break_ties_randomly(action_values, *mcts_manager);
    }

    shared_ptr<const Action> MaxUctDNode::recommend_action_most_visited() const {
        unordered_map<shared_ptr<const Action>, int> visit_counts;

        for (shared_ptr<const Action> action : *actions) {
            if (!has_child_node(action)) {
                continue;
            }
            visit_counts[action] = get_child_node(action)->num_visits;
        }

        if (visit_counts.empty()) {
            int index = mcts_manager->get_rand_int(0, actions->size());
            return actions->at(index);
        }

        return helper::get_max_key_break_ties_randomly(visit_counts, *mcts_manager);
    }

    shared_ptr<const Action> MaxUctDNode::recommend_action(MctsEnvContext& ctx) const {
        UctManager& manager = static_cast<UctManager&>(*mcts_manager);
        (void) ctx;
        if (manager.recommend_most_visited) {
            return recommend_action_most_visited();
        }
        return recommend_action_best_empirical();
    }

    void MaxUctDNode::backup_max_return() {
        num_backups++;
        if (!mcts_manager->mcts_env->is_sink_state_itfc(state)) {
            double best_child_return = is_opponent()
                ? numeric_limits<double>::infinity()
                : -numeric_limits<double>::infinity();
            bool found_child = false;
            for (shared_ptr<const Action> action : *actions) {
                if (has_child_node(action)) {
                    shared_ptr<const MaxUctCNode> child_node = get_child_node(action);
                    const double child_return = child_node->max_return;
                    if (is_opponent()) {
                        best_child_return = min(best_child_return, child_return);
                    } else {
                        best_child_return = max(best_child_return, child_return);
                    }
                    found_child = true;
                }
            }
            if (found_child) {
                max_return = best_child_return;
            }
        }
    }

    void MaxUctDNode::backup(
        const vector<double>& trial_rewards_before_node,
        const vector<double>& trial_rewards_after_node,
        const double trial_cumulative_return_after_node,
        const double trial_cumulative_return,
        MctsEnvContext& ctx)
    {
        (void) trial_rewards_before_node;
        (void) trial_rewards_after_node;
        (void) trial_cumulative_return_after_node;
        (void) trial_cumulative_return;
        (void) ctx;
        backup_max_return();
    }

    shared_ptr<MaxUctCNode> MaxUctDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<MaxUctCNode>(
            static_pointer_cast<UctManager>(mcts_manager),
            state,
            action,
            decision_depth,
            decision_timestep,
            static_pointer_cast<const MaxUctDNode>(shared_from_this()));
    }

    string MaxUctDNode::get_pretty_print_val() const {
        stringstream ss;
        ss << max_return;
        return ss.str();
    }
}

namespace mcts {
    shared_ptr<MaxUctCNode> MaxUctDNode::create_child_node(shared_ptr<const Action> action) {
        shared_ptr<MctsCNode> new_child = MctsDNode::create_child_node_itfc(action);
        return static_pointer_cast<MaxUctCNode>(new_child);
    }

    bool MaxUctDNode::has_child_node(shared_ptr<const Action> action) const {
        return MctsDNode::has_child_node_itfc(static_pointer_cast<const Action>(action));
    }

    shared_ptr<MaxUctCNode> MaxUctDNode::get_child_node(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<MctsCNode> new_child = MctsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<MaxUctCNode>(new_child);
    }
}

namespace mcts {
    void MaxUctDNode::visit_itfc(MctsEnvContext& ctx) {
        visit(ctx);
    }

    shared_ptr<const Action> MaxUctDNode::select_action_itfc(MctsEnvContext& ctx) {
        return select_action(ctx);
    }

    shared_ptr<const Action> MaxUctDNode::recommend_action_itfc(MctsEnvContext& ctx) const {
        return recommend_action(ctx);
    }

    void MaxUctDNode::backup_itfc(
        const vector<double>& trial_rewards_before_node,
        const vector<double>& trial_rewards_after_node,
        const double trial_cumulative_return_after_node,
        const double trial_cumulative_return,
        MctsEnvContext& ctx)
    {
        backup(
            trial_rewards_before_node,
            trial_rewards_after_node,
            trial_cumulative_return_after_node,
            trial_cumulative_return,
            ctx);
    }

    shared_ptr<MctsCNode> MaxUctDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<MaxUctCNode> child_node = create_child_node_helper(action);
        return static_pointer_cast<MctsCNode>(child_node);
    }
}
