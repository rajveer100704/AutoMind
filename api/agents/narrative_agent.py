# api/agents/narrative_agent.py

class NarrativeAgent:
    """
    Generates human-readable narrative for reports.
    """

    def generate(self, metrics: dict, task: str) -> str:

        if not metrics or "error" in metrics:
            return "Evaluation failed. Narrative unavailable."

        if task == "classification":
            return (
                f"The classification model achieved an accuracy of {metrics.get('accuracy',0)*100:.2f}%. "
                f"F1 score was {metrics.get('f1',0):.3f}, "
                f"precision {metrics.get('precision',0):.3f}, "
                f"recall {metrics.get('recall',0):.3f}."
            )

        if task == "regression":
            return (
                f"The regression model achieved an R² of {metrics.get('r2',0):.3f}. "
                f"RMSE was {metrics.get('rmse',0):.3f}, and "
                f"MAE was {metrics.get('mae',0):.3f}."
            )

        return "Narrative unavailable for this task."
