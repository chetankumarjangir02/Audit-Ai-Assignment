
from typing import Any, Dict, Callable, List
import traceback

#  LangGraph If not available using fallback simple graph.
try:
    from langgraph import StateGraph, Node
    LANGGRAPH_AVAILABLE = True
except Exception:
    StateGraph = None
    Node = None
    LANGGRAPH_AVAILABLE = False


class SimpleNode:
    def __init__(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.name = name
        self.fn = fn

    def run(self, state: Dict[str, Any]):
        return self.fn(state) or state


class SimpleGraph:
    def __init__(self):
        self.nodes: List[SimpleNode] = []

    def add_node(self, node: SimpleNode):
        self.nodes.append(node)

    def invoke(self, state: Dict[str, Any]):
        try:
            for node in self.nodes:
                state = node.run(state)
            return state
        except Exception as e:
            traceback.print_exc()
            state["error"] = str(e)
            return state


def normalize_executor_output(state: Dict[str, Any]):
    # Ensure parsed_data is always a dict with "transactions" list
    if "parsed_data" in state:
        parsed = state.get("parsed_data")
        if isinstance(parsed, dict) and "transactions" in parsed:
            state["parsed_data"] = parsed
        else:
            state["parsed_data"] = {"transactions": parsed if parsed is not None else []}

    # Ensure labeled_data is a list (not nested dict)
    if "labeled_data" in state:
        labeled = state.get("labeled_data")
        if isinstance(labeled, dict) and "transactions" in labeled:
            state["labeled_data"] = labeled["transactions"]
        # if it's None or other, leave as-is or coerce to empty list
        if state.get("labeled_data") is None:
            state["labeled_data"] = []

    return state


def build_workflow(planner, executor, reviewer, labeler, rag, qa_generator, fine_tuner):
    """
    Build a small workflow: planner -> executor -> reviewer -> labeler
    The executor node runs each subtask from the plan (planner.process sets plan).
    """

    # LANGGRAPH version (if available)
    if LANGGRAPH_AVAILABLE:
        try:
            graph = StateGraph(name="AuditWorkflow")

            planner_node = Node("planner", lambda s: planner.process(s))

            # Executor node runs subtasks produced by planner
            def executor_step(state):
                plan = state.get("plan", {}).get("subtasks", [])
                for subtask in plan:
                    state["current_task"] = subtask.get("task", "")
                    state = executor.process(state)
                # Normalize executor outputs (parsed_data / labeled_data / summary)
                state = normalize_executor_output(state)
                return state

            executor_node = Node("executor", executor_step)
            reviewer_node = Node("reviewer", lambda s: reviewer.process(s))

            def labeler_step(state):
                # run labeler only if needed
                if state.get("needs_labeling") or ("label" in state.get("user_input", "").lower() or "process" in state.get("user_input", "").lower()):
                    state = labeler.process(state)
                return state

            labeler_node = Node("labeler", labeler_step)

            graph.add_node(planner_node)
            graph.add_node(executor_node)
            graph.add_node(reviewer_node)
            graph.add_node(labeler_node)

            graph.add_edge("planner", "executor")
            graph.add_edge("executor", "reviewer")
            graph.add_edge("reviewer", "labeler")

            return graph
        except Exception:
            print(" LangGraph import error. Falling back to SimpleGraph.")

    # SimpleGraph fallback
    simple = SimpleGraph()

    # planner
    simple.add_node(SimpleNode("planner", lambda s: planner.process(s)))

    # executor wrapper that runs plan subtasks and normalizes outputs
    def simple_executor_wrapper(state):
        plan = state.get("plan", {}).get("subtasks", [])
        for subtask in plan:
            state["current_task"] = subtask.get("task", "")
            state = executor.process(state)
        state = normalize_executor_output(state)
        return state

    simple.add_node(SimpleNode("executor", simple_executor_wrapper))

    # reviewer
    simple.add_node(SimpleNode("reviewer", lambda s: reviewer.process(s)))

    # labeler wrapper
    def simple_labeler(state):
        if state.get("needs_labeling") or ("label" in state.get("user_input", "").lower() or "process" in state.get("user_input", "").lower()):
            state = labeler.process(state)
        return state

    simple.add_node(SimpleNode("labeler", simple_labeler))

    return simple



