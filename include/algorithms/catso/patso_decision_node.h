#pragma once

#include "algorithms/catso/catso_decision_node.h"
#include "algorithms/catso/patso_manager.h"

#include <memory>

namespace mcts {
    class PatsoCNode;

    class PatsoDNode : public CatsoDNode {
        friend PatsoCNode;

        public:
            PatsoDNode(
                std::shared_ptr<PatsoManager> mcts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const PatsoCNode> parent=nullptr);

            virtual ~PatsoDNode() = default;

        protected:
            std::shared_ptr<CatsoCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

        public:
            std::shared_ptr<PatsoCNode> create_child_node(std::shared_ptr<const Action> action);
            bool has_child_node(std::shared_ptr<const Action> action) const;
            std::shared_ptr<PatsoCNode> get_child_node(std::shared_ptr<const Action> action) const;
            virtual std::shared_ptr<MctsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
    };
}
