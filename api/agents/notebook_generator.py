# api/agents/notebook_generator.py

import nbformat as nbf
import os


class NotebookGenerator:

    def generate_notebook(self, path: str, summary: dict, explainability_images=None):
        nb = nbf.v4.new_notebook()
        cells = []

        cells.append(nbf.v4.new_markdown_cell(f"# AutoMind Notebook — Run {summary.get('run_id')}"))

        cells.append(nbf.v4.new_markdown_cell(
            "## Dataset Overview\n"
            f"- Rows: {summary.get('rows')}\n"
            f"- Columns: {len(summary.get('columns', []))}\n"
            f"- Target: **{summary.get('target')}**\n"
            f"- Task: **{summary.get('task_type')}**"
        ))

        metrics = summary.get("metrics")
        if metrics:
            md = "## Evaluation Metrics\n"
            for k, v in metrics.items():
                md += f"- **{k}**: {v}\n"
            cells.append(nbf.v4.new_markdown_cell(md))

        if explainability_images:
            for i, enc in enumerate(explainability_images):
                cells.append(nbf.v4.new_markdown_cell(
                    f"### SHAP Plot {i+1}\n"
                    f"<img src='data:image/png;base64,{enc}' width='700'/>"
                ))

        cells.append(nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "df = pd.read_csv('your_dataset.csv')\n"
            "df.head()"
        ))

        nb["cells"] = cells

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)

        return path
