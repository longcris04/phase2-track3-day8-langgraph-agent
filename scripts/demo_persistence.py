"""Extension demo: durable HITL with crash-resume via SQLite checkpointer.

Demonstrates that the graph can pause at the human-approval gate using
``langgraph.types.interrupt()``, survive a *process restart* (we throw away the
graph object and rebuild it from the on-disk SQLite checkpoint), and then resume
to completion with the human's decision.

Run:
    LANGGRAPH_INTERRUPT=true python scripts/demo_persistence.py
"""

from __future__ import annotations

import os
from pathlib import Path

from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state

DB = "outputs/demo_resume.sqlite"


def main() -> None:
    os.environ["LANGGRAPH_INTERRUPT"] = "true"
    # Start clean so the demo is reproducible.
    for suffix in ("", "-wal", "-shm"):
        Path(DB + suffix).unlink(missing_ok=True)

    scenario = Scenario(
        id="demo_resume",
        query="Refund this customer and send confirmation email",
        expected_route=Route.RISKY,
        requires_approval=True,
    )
    state = initial_state(scenario)
    cfg = {"configurable": {"thread_id": state["thread_id"]}}

    print("=== Process A: run until the approval interrupt ===")
    graph_a = build_graph(checkpointer=build_checkpointer("sqlite", DB))
    result_a = graph_a.invoke(state, config=cfg)
    assert "__interrupt__" in result_a, "expected the graph to pause at approval"
    payload = result_a["__interrupt__"][0].value
    print("  paused. route so far:", graph_a.get_state(cfg).values.get("route"))
    print("  interrupt asked:", payload["question"])

    # Simulate a crash: drop every in-memory object. Only SQLite on disk remains.
    del graph_a, result_a

    print("\n=== Process B: rebuild from SQLite and resume with the human decision ===")
    graph_b = build_graph(checkpointer=build_checkpointer("sqlite", DB))
    recovered = graph_b.get_state(cfg)
    print("  recovered pending node:", recovered.next, "| attempt:", recovered.values.get("attempt"))
    human = {"approved": True, "comment": "Refund approved by supervisor."}
    final = graph_b.invoke(Command(resume=human), config=cfg)
    print("  resumed -> route:", final["route"], "| approval:", final["approval"])
    print("  final answer:", final["final_answer"][:80])

    history = list(graph_b.get_state_history(cfg))
    print("\nRESUME_SUCCESS:", bool(final.get("final_answer")) and final["approval"]["approved"])
    print("checkpoints in history:", len(history))


if __name__ == "__main__":
    main()
