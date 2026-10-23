# src/agents/run_agent.py
import warnings

# Ignore Google GenAI automatic function calling warnings in CLI view
warnings.filterwarnings("ignore", category=UserWarning, module="google_genai.models")

from src.agents.graph import financial_agent
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


def run_agent(question: str) -> dict:
    """Run the compiled agent graph on a single question and return a summary."""
    logger.info(f"Running agent for: '{question}'")

    try:
        final_state = financial_agent.invoke({"question": question})

        # Ensure answer is always extracted cleanly as a string if it got wrapped
        raw_answer = final_state.get("final_answer", "")
        if isinstance(raw_answer, dict):
            answer_text = raw_answer.get("text", str(raw_answer))
        else:
            answer_text = str(raw_answer)

        return {
            "question": question,
            "task_type": final_state.get("task_type"),
            "banks": final_state.get("banks"),
            "answer": answer_text,
            "confidence": final_state.get("confidence"),
            "errors": final_state.get("errors", []),
        }
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {
            "question": question,
            "answer": "Agent execution failed due to an internal error.",
            "confidence": "low",
            "errors": [str(e)],
        }


if __name__ == "__main__":
    print("Financial Agent CLI — type 'exit' to quit\n")
    while True:
        try:
            query = input("Query: ").strip()
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue

            result = run_agent(query)
            print(f"\n[{result['task_type']} | {result['banks']} | {result['confidence']}]")
            print("-" * 50)
            print(result["answer"])
            print("-" * 50 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
