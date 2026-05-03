#pragma once

#include "algorithms/uct/uct_logger.h"
#include "algorithms/uct/uct_manager.h"
#include "algorithms/common/empirical_cvar.h"
#include "mcts_chance_node.h"
#include "mcts_env_context.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace mcts {
    class MaxUctDNode;

    /**
     * Implementation of MaxUCT-style chance nodes in the Mcts schema.
     *
     * This is structurally copied from UctCNode, but inherits directly from MctsCNode so that
     * the backup behaviour can diverge cleanly from standard UCT.
     */
    class MaxUctCNode : public MctsCNode {
        friend MaxUctDNode;
        friend UctLogger;

        protected:
            int num_backups;
            double max_return;
            std::shared_ptr<StateDistr> next_state_distr;
            bool track_empirical_cvar;
            std::vector<double> empirical_return_samples;

            std::shared_ptr<const State> sample_observation_random();

            // TODO: Implement max-return backup logic here.
            void backup_max_return(const double trial_return_after_node);

        public:
            MaxUctCNode(
                std::shared_ptr<UctManager> mcts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const MaxUctDNode> parent=nullptr);

            void visit(MctsEnvContext& ctx);
            std::shared_ptr<const State> sample_observation(MctsEnvContext& ctx);
            void backup(
                const std::vector<double>& trial_rewards_before_node,
                const std::vector<double>& trial_rewards_after_node,
                const double trial_cumulative_return_after_node,
                const double trial_cumulative_return,
                MctsEnvContext& ctx);

        protected:
            std::shared_ptr<MaxUctDNode> create_child_node_helper(std::shared_ptr<const State> observation) const;
            virtual std::string get_pretty_print_val() const;

        public:
            virtual ~MaxUctCNode() = default;
            double get_cvar_value_at(double alpha) const;

            std::shared_ptr<MaxUctDNode> create_child_node(std::shared_ptr<const State> observation);
            bool has_child_node(std::shared_ptr<const State> observation) const;
            std::shared_ptr<MaxUctDNode> get_child_node(std::shared_ptr<const State> observation) const;

            virtual void visit_itfc(MctsEnvContext& ctx);
            virtual std::shared_ptr<const Observation> sample_observation_itfc(MctsEnvContext& ctx);
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node,
                const std::vector<double>& trial_rewards_after_node,
                const double trial_cumulative_return_after_node,
                const double trial_cumulative_return,
                MctsEnvContext& ctx);

            virtual std::shared_ptr<MctsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation,
                std::shared_ptr<const State> next_state=nullptr) const;
    };
}
