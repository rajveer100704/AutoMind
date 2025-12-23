# api/agents/master_agent.py

import os
import uuid
import json
from datetime import datetime
from typing import Dict, Any

from .data_loader import DataLoaderAgent
from .target_detector import TargetDetectorAgent
from .problem_type_detector import ProblemTypeDetectorAgent
from .eda_agent import EDAAgent
from .preprocessing_agent import PreprocessingAgent
from .feature_engineering_agent import FeatureEngineeringAgent
from .advanced_feature_engineering import AdvancedFeatureEngineeringAgent
from .sampling_policy import SamplingPolicyAgent
from .model_agent import ModelTrainingAgent
from .timeseries_agent import TimeSeriesAgent
from .evaluation_agent import EvaluationAgent
from .explainability_agent import ExplainabilityAgent
from .narrative_agent import NarrativeAgent
from .notebook_generator import NotebookGenerator
from .report_agent import ReportAgent
from .llm_agent import LLMReasoner

from api.monitoring import emit_event

class AutoMindMasterAgent:
    """
    Orchestrates the complete AutoMind DS-Agent pipeline.
    """

    def __init__(self, run_id=None, template_dir="api/templates"):
        self.run_id = run_id or (datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
        self.template_dir = template_dir

        os.makedirs("reports", exist_ok=True)
        os.makedirs("artifacts", exist_ok=True)
        os.makedirs("mlruns", exist_ok=True)

        self.llm = LLMReasoner()
        self.report_agent = ReportAgent(template_dir=template_dir)

    def _log(self, step: str, status: str, details: Dict[str, Any] = None):
        emit_event(self.run_id, step, status, details or {})

    # ------------------------------------------------------------

    def run_pipeline(
        self,
        df,
        autodetect_target=True,
        target_override=None,
        tune_rounds=10,
        advanced_fe=False,
        sample_frac=None,
        notebook=True
    ):

        self._log("pipeline", "start", {"run_id": self.run_id})
        # ========================== 1. Load Data ==========================
        self._log("load_data", "start")

        loader = DataLoaderAgent()
        try:
            df = loader.load(df)
        except Exception as e:
            self._log("load_data", "error", {"error": str(e)})
            raise

        self._log("load_data", "complete", {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1])
        })

        # ======================= 2. LLM: EDA Plan ========================
        self._log("llm_eda_plan", "start")
        plan = self.llm.think(
            "EDA",
            {"columns": list(df.columns)[:30], "rows": int(df.shape[0])},
            "Suggest EDA checks and visuals."
        )
        self._log("llm_eda_plan", "complete", {"plan": plan})

        # ==================== 3. Target Detection ========================
        self._log("target_detect", "start")
        detector = TargetDetectorAgent()
        target = detector.detect(df, target_override, autodetect_target)
        self._log("target_detect", "complete", {"target": target})

        # ================== 4. Problem Type Detection ====================
        self._log("problem_detect", "start")
        type_agent = ProblemTypeDetectorAgent()
        task_type = type_agent.detect(df, target)
        self._log("problem_detect", "complete", {"task": task_type})

        # ============================ 5. EDA =============================
        self._log("eda", "start")
        eda = EDAAgent()
        eda_summary, eda_images = eda.analyze(df, target)
        self._log("eda", "complete", {
            "summary": list(eda_summary.keys()),
            "img_count": len(eda_images)
        })

        # ======================== 6. Preprocessing =======================
        self._log("preprocess", "start")
        prep = PreprocessingAgent()
        try:
            df_pre = prep.process(df, target)
            self._log("preprocess", "complete", {
                "rows": int(df_pre.shape[0]),
                "cols": int(df_pre.shape[1])
            })
        except Exception as e:
            self._log("preprocess", "error", {"error": str(e)})
            df_pre = df.copy()

        # ===================== 7. Basic FE ===============================
        self._log("feature_engineering", "start")
        fe = FeatureEngineeringAgent()
        try:
            df_fe = fe.transform(df_pre, target)
        except Exception:
            df_fe = df_pre.copy()
        self._log("feature_engineering", "complete", {"cols": int(df_fe.shape[1])})

        # ==================== 8. Advanced FE (Optional) =================
        if advanced_fe:
            self._log("advanced_fe", "start")
            adv = AdvancedFeatureEngineeringAgent()
            try:
                df_fe = adv.enhance(df_fe, target)
                self._log("advanced_fe", "complete", {"cols": int(df_fe.shape[1])})
            except Exception as e:
                self._log("advanced_fe", "error", {"error": str(e)})

        # ========================= 9. Sampling ===========================
        if sample_frac:
            self._log("sampling", "start")
            sampler = SamplingPolicyAgent()
            try:
                df_fe = sampler.sample(df_fe, target)
                self._log("sampling", "complete", {"rows": int(df_fe.shape[0])})
            except Exception as e:
                self._log("sampling", "error", {"error": str(e)})

        # ========================= 10. Modeling ==========================
        self._log("model_training", "start")
        leaderboard = []
        model = None

        try:
            if task_type == "timeseries":
                ts = TimeSeriesAgent()
                model, leaderboard = ts.fit(df_fe, target)
            else:
                trainer = ModelTrainingAgent(self.run_id)
                model, leaderboard = trainer.train(df_fe, target, task_type, tune_rounds)
        except Exception as e:
            self._log("model_training", "error", {"error": str(e)})
        self._log("model_training", "complete", {"leaderboard_len": len(leaderboard) if leaderboard else 0})

        # Normalize leaderboard into serializable structure
        try:
            # If pandas DataFrame -> convert to records
            import pandas as pd
            if isinstance(leaderboard, pd.DataFrame):
                lb_serial = leaderboard.to_dict(orient="records")
            else:
                lb_serial = leaderboard
        except Exception:
            lb_serial = leaderboard

        # ======================== 11. Evaluation =========================
        self._log("evaluation", "start")
        evaluator = EvaluationAgent()
        try:
            metrics = evaluator.evaluate(model, df_fe, target, task_type)
        except Exception as e:
            metrics = {"error": f"Evaluation failed: {str(e)}"}
            self._log("evaluation", "error", {"error": str(e)})
        self._log("evaluation", "complete", {"metrics": metrics})

        # ========================= 12. Explainability ====================
        self._log("explainability", "start")
        explainer = ExplainabilityAgent()
        try:
            shap_images = explainer.explain(model, df_fe, target)
            self._log("explainability", "complete", {"shap": len(shap_images)})
        except Exception as e:
            shap_images = []
            self._log("explainability", "error", {"error": str(e)})

        # ====================== 13. Narrative (LLM) ======================
        self._log("narrative", "start")
        narrative_agent = NarrativeAgent()
        try:
            narrative = narrative_agent.generate(metrics, task_type)
        except Exception:
            narrative = "Narrative generation failed or unavailable."
        self._log("narrative", "complete", {"preview": str(narrative)[:200]})

        # ==================== 14. Notebook Generation ====================
        nb_path = None
        if notebook:
            self._log("notebook", "start")
            nb_gen = NotebookGenerator()
            try:
                nb_path = nb_gen.generate_notebook(
                    f"reports/notebook_{self.run_id}.ipynb",
                    {
                        "run_id": self.run_id,
                        "rows": df.shape[0],
                        "columns": list(df.columns),
                        "target": target,
                        "task_type": task_type,
                        "metrics": metrics
                    },
                    shap_images
                )
                self._log("notebook", "complete", {"path": nb_path})
            except Exception as e:
                nb_path = None
                self._log("notebook", "error", {"error": str(e)})

        # ======================== 15. Report =============================
        self._log("report_build", "start")
        report_path = f"reports/report_{self.run_id}.html"
        try:
            # Ensure leaderboard is safe for template (list of dict)
            self.report_agent.generate_html(report_path, {
                "run_id": self.run_id,
                "meta": {
                    "rows": df.shape[0],
                    "columns": list(df.columns),
                    "target": target,
                    "task_type": task_type
                },
                "metrics": metrics,
                "leaderboard": lb_serial,
                "best_model": str(model),
                "narrative": narrative,
                "evaluation_images": eda_images,
                "shap_images": shap_images
            })
            self._log("report_build", "complete", {"path": report_path})
        except Exception as e:
            self._log("report_build", "error", {"error": str(e)})

        # ==================== 16. ZIP Artifact Bundle ====================
        self._log("artifact", "start")
        artifact_path = f"artifacts/artifact_{self.run_id}.zip"
        try:
            import zipfile
            with zipfile.ZipFile(artifact_path, "w") as z:
                if os.path.exists(report_path):
                    z.write(report_path, os.path.basename(report_path))
                if nb_path and os.path.exists(nb_path):
                    z.write(nb_path, os.path.basename(nb_path))
            self._log("artifact", "complete", {"path": artifact_path})
        except Exception as e:
            self._log("artifact", "error", {"error": str(e)})

        # ==================== 17. Append run history for quick UI =================
        history_entry = {
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "task_type": task_type,
        }
        try:
            hist_path = "run_history.json"
            if os.path.exists(hist_path):
                old = json.load(open(hist_path, "r", encoding="utf-8")) or []
            else:
                old = []
            old.append(history_entry)
            json.dump(old, open(hist_path, "w", encoding="utf-8"), indent=2)
        except Exception:
            pass

        self._log("pipeline", "complete", {"run_id": self.run_id})
        return {
            "run_id": self.run_id,
            "report": report_path,
            "notebook": nb_path,
            "artifact": artifact_path,
            "metrics": metrics,
            "leaderboard": lb_serial,
            "narrative": narrative
        }
