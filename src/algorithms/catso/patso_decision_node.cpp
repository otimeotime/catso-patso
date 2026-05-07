#include "algorithms/catso/patso_decision_node.h"

#include "algorithms/catso/patso_chance_node.h"

using namespace std;

namespace mcts {
    PatsoDNode::PatsoDNode(
        shared_ptr<PatsoManager> mcts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const PatsoCNode> parent) :
            CatsoDNode(mcts_manager, state, decision_depth, decision_timestep, parent)
    {
    }

    shared_ptr<CatsoCNode> PatsoDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<PatsoCNode>(
            static_pointer_cast<PatsoManager>(mcts_manager),
            state,
            action,
            decision_depth,
            decision_timestep,
            static_pointer_cast<const PatsoDNode>(shared_from_this()));
    }
}

namespace mcts {
    shared_ptr<PatsoCNode> PatsoDNode::create_child_node(shared_ptr<const Action> action) {
        shared_ptr<MctsCNode> new_child = MctsDNode::create_child_node_itfc(action);
        return static_pointer_cast<PatsoCNode>(new_child);
    }

    bool PatsoDNode::has_child_node(shared_ptr<const Action> action) const {
        return MctsDNode::has_child_node_itfc(static_pointer_cast<const Action>(action));
    }

    shared_ptr<PatsoCNode> PatsoDNode::get_child_node(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<MctsCNode> new_child = MctsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<PatsoCNode>(new_child);
    }

    shared_ptr<MctsCNode> PatsoDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<CatsoCNode> child_node = create_child_node_helper(action);
        return static_pointer_cast<MctsCNode>(child_node);
    }
}
