# api/agents/report_agent.py

import os
import base64
from jinja2 import Environment, FileSystemLoader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO


class ReportAgent:

    def __init__(self, template_dir):
        env = Environment(loader=FileSystemLoader(template_dir))
        self.template = env.get_template("report_template.html")

    def generate_html(self, output_path, context: dict):
        html = self.template.render(**context)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    def generate_pdf(self, output_path, context: dict):
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(f"AutoMind Report — Run {context.get('run_id')}", styles['Title']))
        story.append(Spacer(1, 12))

        meta = context.get("meta", {})
        story.append(Paragraph(f"Task: {meta.get('task_type')}", styles['BodyText']))
        story.append(Paragraph(f"Rows: {meta.get('rows')}", styles['BodyText']))
        story.append(Spacer(1, 12))

        if context.get("narrative"):
            story.append(Paragraph("Narrative Summary:", styles['Heading2']))
            story.append(Paragraph(context["narrative"], styles['BodyText']))
            story.append(Spacer(1, 12))

        def add_b64_image(encoded):
            try:
                data = base64.b64decode(encoded)
                img = RLImage(BytesIO(data), width=450)
                story.append(img)
                story.append(Spacer(1, 12))
            except:
                pass

        for img in context.get("shap_images", []):
            add_b64_image(img)

        for img in context.get("evaluation_images", []):
            add_b64_image(img)

        doc = SimpleDocTemplate(output_path, pagesize=A4)
        doc.build(story)

        return output_path
