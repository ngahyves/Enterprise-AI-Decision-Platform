# src/agents/tools/calculator.py
from langchain_core.tools import tool
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


@tool
def calculate_percentage_change(old_value: float, new_value: float) -> dict:
    """
    Calculates the percentage change between an old financial value and a new one.
    Useful for year-over-year comparisons (revenue growth, expense changes).
    Args:
        old_value: The baseline (e.g., from 2024).
        new_value: The recent (e.g., from 2025).
    Returns:
        A dict with keys: percentage (float), direction ('increase'|'decrease'|'flat'),
        absolute_change (float), error (str, only if the calculation failed).
    """
    if old_value == 0:
        logger.warning(f"Percentage change with zero baseline: old={old_value}, new={new_value}")
        return {
            "error": "baseline_zero",
            "percentage": 0.0,
            "direction": "flat",
            "absolute_change": 0.0,
        }

    change = new_value - old_value
    percentage = (change / abs(old_value)) * 100
    direction = "increase" if percentage > 0 else "decrease" if percentage < 0 else "flat"

    result = {
        "percentage": round(percentage, 2),
        "direction": direction,
        "absolute_change": round(change, 2),
    }
    logger.info(f"Calculator: old={old_value}, new={new_value} -> {result}")
    return result


@tool
def calculate_sum(values: list[float]) -> dict:
    """
    Sums a list of financial figures.
    Args:
        values: List of numerical values to sum.
    Returns:
        A dict with keys: total (float), count (int).
    """
    total = sum(values)
    logger.info(f"Calculator sum: {len(values)} items -> {total}")
    return {"total": round(total, 2), "count": len(values)}
