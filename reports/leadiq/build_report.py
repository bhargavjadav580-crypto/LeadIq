"""Generate the LeadIQ academic project report PDF (mirrors the LibraScan template)."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                                Table, TableStyle, Image, PageBreak, KeepTogether, ListFlowable, ListItem)
from reportlab.platypus.flowables import HRFlowable
from reportlab.pdfgen import canvas as canvaslib
from PIL import Image as PILImage

AST = "/app/reports/leadiq/assets"
OUTPUT = "/app/LeadIQ_Project_Report.pdf"

INK = colors.HexColor("#1e1b2e")
VIO = colors.HexColor("#5b3df2")
VIO_D = colors.HexColor("#6d5bd0")
GREY = colors.HexColor("#555068")
LIGHT = colors.HexColor("#ece9f5")
BORDER = colors.HexColor("#b9b4cc")
HEAD_L = "LeadIQ: Explainable AI Lead Scoring & Prioritization System"
HEAD_R = "Machine Learning (BE05000211)"

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm

# ----------------------------------------------------------------------------- styles
ss = getSampleStyleSheet()
def mk(name, **kw):
    return ParagraphStyle(name, parent=ss["Normal"], **kw)

body = mk("body", fontName="Times-Roman", fontSize=11, leading=15.5,
          alignment=TA_JUSTIFY, textColor=INK, spaceAfter=7)
bullet = mk("bullet", fontName="Times-Roman", fontSize=11, leading=15,
            textColor=INK, alignment=TA_JUSTIFY)
h1 = mk("h1", fontName="Times-Bold", fontSize=17, leading=21, textColor=INK,
        spaceBefore=4, spaceAfter=12)
h2 = mk("h2", fontName="Times-Bold", fontSize=13, leading=17, textColor=INK,
        spaceBefore=11, spaceAfter=5)
h3 = mk("h3", fontName="Times-Bold", fontSize=11.5, leading=15, textColor=VIO_D,
        spaceBefore=8, spaceAfter=3)
cap = mk("cap", fontName="Times-Italic", fontSize=9.5, leading=12.5, textColor=GREY,
         alignment=TA_CENTER, spaceBefore=4, spaceAfter=12)
cell = mk("cell", fontName="Times-Roman", fontSize=9.5, leading=12.5, textColor=INK)
cellb = mk("cellb", fontName="Times-Bold", fontSize=9.5, leading=12.5, textColor=INK)
cellh = mk("cellh", fontName="Times-Bold", fontSize=9.5, leading=12.5, textColor=colors.white)
code = mk("code", fontName="Courier", fontSize=9, leading=13, textColor=INK,
          backColor=colors.HexColor("#f4f2fb"), borderPadding=6, spaceBefore=4, spaceAfter=10)

# cover styles
cv_center = mk("cv_center", fontName="Times-Bold", fontSize=15, leading=24,
               alignment=TA_CENTER, textColor=INK)
cv_sub = mk("cv_sub", fontName="Times-Bold", fontSize=13, leading=20, alignment=TA_CENTER, textColor=INK)
cv_title = mk("cv_title", fontName="Times-Bold", fontSize=22, leading=28, alignment=TA_CENTER, textColor=INK)
cv_desc = mk("cv_desc", fontName="Times-Italic", fontSize=13.5, leading=19, alignment=TA_CENTER, textColor=INK)
cv_label = mk("cv_label", fontName="Times-Bold", fontSize=12, leading=18, textColor=INK)
cv_val = mk("cv_val", fontName="Times-Roman", fontSize=12, leading=18, textColor=INK)

story = []

def para(t, s=body): story.append(Paragraph(t, s))
def sp(h=6): story.append(Spacer(1, h))

def bullets(items, s=bullet):
    story.append(ListFlowable(
        [ListItem(Paragraph(x, s), leftIndent=6, value="\u2022") for x in items],
        bulletType="bullet", bulletColor=VIO, bulletFontSize=8, leftIndent=14, spaceAfter=8))

def pic(name, width_mm, caption):
    path = os.path.join(AST, name)
    iw, ih = PILImage.open(path).size
    w = width_mm * mm
    h = w * ih / iw
    maxh = 205 * mm
    if h > maxh:
        h = maxh; w = h * iw / ih
    img = Image(path, width=w, height=h)
    img.hAlign = "CENTER"
    story.append(KeepTogether([img, Paragraph(caption, cap)]))

def framed_pic(name, width_mm, caption):
    """Screenshot with a thin border."""
    path = os.path.join(AST, name)
    iw, ih = PILImage.open(path).size
    w = width_mm * mm
    h = w * ih / iw
    maxh = 150 * mm
    if h > maxh:
        h = maxh; w = h * iw / ih
    img = Image(path, width=w, height=h)
    t = Table([[img]], colWidths=[w])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    t.hAlign = "CENTER"
    story.append(KeepTogether([t, Paragraph(caption, cap)]))

def table(data, col_mm, caption=None, header=True, align=None, font=9.5):
    cols = [c * mm for c in col_mm]
    def wrap(v, sty):
        return Paragraph(str(v), sty) if not isinstance(v, Paragraph) else v
    rows = []
    for i, r in enumerate(data):
        sty = cellh if (header and i == 0) else cell
        rows.append([wrap(c, sty) for c in r])
    t = Table(rows, colWidths=cols, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f6f4fc")]),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), VIO_D),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    if align:
        for col, a in align.items():
            style.append(("ALIGN", (col, 0), (col, -1), a))
    t.setStyle(TableStyle(style))
    t.hAlign = "CENTER"
    flows = [t]
    if caption:
        flows.append(Paragraph(caption, cap))
    else:
        flows.append(Spacer(1, 10))
    story.append(KeepTogether(flows) if sum(len(str(c)) for r in data for c in r) < 700 else t)
    if not caption and sum(len(str(c)) for r in data for c in r) >= 700:
        story.append(Spacer(1, 10))

# ============================================================= COVER PAGE
def cover():
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("L.D. College of Engineering, Ahmedabad", cv_center))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Department of Artificial Intelligence and Machine Learning", cv_sub))
    story.append(Paragraph("Machine Learning (BE05000211)", cv_sub))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Project Report", mk("pr", fontName="Times-Bold", fontSize=13, alignment=TA_CENTER, textColor=INK)))
    story.append(Spacer(1, 16 * mm))
    story.append(Paragraph("REPORT ON LEADIQ", cv_title))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Explainable AI Lead Scoring &amp; Prioritization System using Machine Learning, "
                           "Text Embeddings and SHAP", cv_desc))
    story.append(Spacer(1, 16 * mm))
    story.append(Paragraph("Submitted By:", cv_label))
    story.append(Spacer(1, 4))
    cvh = mk("cvh", fontName="Times-Bold", fontSize=10.5, leading=13, textColor=INK, alignment=TA_CENTER)
    students = [
        [Paragraph("Sr. No.", cvh), Paragraph("Name of Student", cvh), Paragraph("Enrollment No.", cvh)],
        [Paragraph("1", cell), Paragraph("Darshil Savaliya", cell), Paragraph("240280152068", cell)],
        [Paragraph("2", cell), Paragraph("Harshil Patel", cell), Paragraph("240280152052", cell)],
        [Paragraph("3", cell), Paragraph("Aranav Mandani", cell), Paragraph("240280152037", cell)],
    ]
    t = Table(students, colWidths=[22 * mm, 85 * mm, 50 * mm], rowHeights=[11 * mm] * 4)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#808080")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9d9d9")),
        ("ALIGN", (0, 0), (0, -1), "CENTER"), ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("LEFTPADDING", (1, 0), (1, -1), 8),
    ]))
    t.hAlign = "CENTER"
    story.append(t)
    story.append(Spacer(1, 14 * mm))
    for lab, val in [("Branch:", "Artificial Intelligence and Machine Learning"),
                     ("Semester:", "5th"),
                     ("Academic Year:", "2026-27"),
                     ("Subject:", "Machine Learning (BE05000211)")]:
        story.append(Paragraph(f'<b>{lab}</b> {val}', cv_val))
        story.append(Spacer(1, 7))
    story.append(Spacer(1, 6))
    story.append(Paragraph('<b>Faculty Name &amp; Sign:</b> ____________________', cv_val))
    story.append(Spacer(1, 7))
    story.append(Paragraph('<b>Date:</b> _________________', cv_val))
    story.append(PageBreak())

# ============================================================= CONTENTS
def contents():
    para("CONTENTS", h1)
    data = [["Sr. No.", "Content", "Page No."],
            ["1", "Introduction", "3"],
            ["2", "Objectives", "5"],
            ["3", "Technologies and Tools", "6"],
            ["4", "System Requirements and Setup", "9"],
            ["5", "System Design", "11"],
            ["6", "Implementation and Working of the System", "16"],
            ["7", "Results and Output Screens", "22"],
            ["8", "Advantages, Limitations and Future Scope", "30"],
            ["9", "Conclusion", "32"],
            ["10", "References", "33"]]
    table(data, [25, 115, 30], align={0: "CENTER", 2: "CENTER"})
    story.append(PageBreak())

cover()
contents()

# ============================================================= 1. INTRODUCTION
para("1. Introduction", h1)
para("1.1 Background", h3)
para("Every sales and outreach team works from a list of possible customers, or <i>leads</i>. "
     "A lead is a business or a person who might eventually buy a product or respond to an offer. "
     "In practice, far more leads are collected than can ever be contacted: a spreadsheet may hold "
     "thousands of rows, yet only a small fraction will ever convert into a real conversation or sale. "
     "The central question for any team is therefore not <i>who is on the list</i>, but <i>who should be "
     "contacted first</i>. Answering that question well saves time, raises conversion rates and keeps "
     "effort focused where it matters.")
para("Traditionally this prioritization is done by hand, using simple rules such as &ldquo;call the "
     "highest-rated businesses first&rdquo;. Such rules ignore most of the information in the data and "
     "cannot weigh many signals at once. Modern machine learning can do better: given historical records "
     "whose outcome is already known, a model can learn which combinations of attributes are associated "
     "with conversion and then score new leads by their estimated probability of converting. LeadIQ, the "
     "system described in this report, turns an uploaded spreadsheet into a ranked, explainable contact "
     "list, with the tagline &ldquo;Your next best lead &mdash; and the why behind it.&rdquo;")
para("1.2 Problem Statement", h3)
para("Scoring leads with machine learning is useful only if the result can be <b>trusted</b>. Three "
     "problems make naive solutions unreliable. First, real lead data is <b>messy and mixed</b>: it "
     "contains numbers, categories, free text and missing values, and it is almost always "
     "<b>imbalanced</b> &mdash; far fewer leads convert than do not. Second, a model that is only "
     "measured on the data it was trained on will look far better than it really is. Third, a bare "
     "probability answers <i>who</i> but never <i>why</i>, so a user cannot judge whether a score is "
     "sensible or act on it with confidence. LeadIQ addresses all three: it cleans and encodes arbitrary "
     "tabular data, it measures every model on an untouched test set with honest uncertainty, and it "
     "explains each prediction with real SHAP values.")
para("1.3 Motivation and Relevance to Machine Learning", h3)
para("The project is a complete, end-to-end example of the <b>Knowledge Discovery in Databases (KDD)</b> "
     "process, the classic framework that turns raw data into usable knowledge through selection, "
     "pre-processing, transformation, data mining and interpretation. LeadIQ maps its nine guided steps "
     "directly onto these KDD stages. The engineering work lies around the models: preparing and cleaning "
     "the input, building features (including sentence-transformer text embeddings), comparing four "
     "classifiers against a no-skill baseline with cross-validation, calibrating probabilities, choosing "
     "an operating threshold, and explaining the winner &mdash; so that model output can be trusted and "
     "acted upon in practice.")
para("1.4 Scope of the Project", h3)
bullets([
    "Upload of any tabular dataset (CSV, TSV, TXT, XLSX or XLS, up to 10&nbsp;MB) with automatic "
    "parsing, sheet selection and column-role detection &mdash; the system is not tied to a fixed schema.",
    "Interactive selection of the target column and the positive outcome, with automatic leakage checks.",
    "Data cleaning with a transparent quality report, and exploratory analysis of outcome patterns.",
    "Automatic feature engineering: numeric scaling, categorical target-encoding, presence/validity flags "
    "for contact fields, and NLP embeddings for free-text columns.",
    "Training and fair comparison of Logistic Regression, Decision Tree, Random Forest and XGBoost against "
    "a dummy baseline, with training-only cross-validation and a reserved test split.",
    "A ten-tab evaluation workspace, probability calibration, an operating threshold, lead scoring into "
    "priority tiers, per-lead SHAP explanations, and CSV / Excel / PDF export.",
])
story.append(PageBreak())

# ============================================================= 2. OBJECTIVES
para("2. Objectives", h1)
para("2.1 Aim", h3)
para("The aim of the project is to build a web-based system that converts an uploaded, outcome-labelled "
     "dataset into a prioritized and explainable contact list, so that a team can decide whom to contact "
     "next with measured confidence, while keeping the user in control of every decision.")
para("2.2 Specific Objectives", h3)
story.append(ListFlowable([ListItem(Paragraph(x, bullet), leftIndent=6) for x in [
    "To accept arbitrary tabular files, parse them robustly, and let the user confirm the target column "
    "and the positive class.",
    "To clean the data &mdash; handling duplicates, missing values and inconsistent types &mdash; and to "
    "report exactly what was changed.",
    "To build a numeric feature matrix automatically, including target-encoded categories and "
    "sentence-transformer embeddings of free-text columns.",
    "To reserve an untouched test set and use training-only cross-validation so that reported performance "
    "is honest and not optimistic.",
    "To train and compare four classifiers against a no-skill baseline and select a winner by "
    "cross-validated average precision (PR-AUC).",
    "To calibrate the winner&rsquo;s probabilities and choose an operating threshold suited to the "
    "class imbalance.",
    "To score leads into percentile scores and priority tiers, and to rank them in a deterministic "
    "contact order.",
    "To explain each individual prediction with real SHAP values and to disclose the model&rsquo;s "
    "limitations honestly.",
    "To export the ranked leads and a summary report as CSV, Excel and PDF.",
]], bulletType="1", leftIndent=18, spaceAfter=7))
story.append(PageBreak())

# ============================================================= 3. TECHNOLOGIES AND TOOLS
para("3. Technologies and Tools", h1)
para("3.1 Supervised Classification and the KDD Process", h3)
para("LeadIQ treats lead scoring as a <b>binary classification</b> problem: each labelled record belongs "
     "to one of two classes (the positive outcome, here &ldquo;converted&rdquo;, or the negative one). A "
     "model learns a function from the input features to a probability of the positive class. The whole "
     "application follows the KDD process &mdash; <i>selection, pre-processing, transformation, data "
     "mining</i> and <i>interpretation / knowledge use</i> &mdash; which gives the nine-step workflow its "
     "academic backbone and keeps each stage separate and inspectable.")
para("3.2 Feature Engineering and Text Embeddings", h3)
para("Numeric columns are scaled; categorical columns are converted to numbers with scikit-learn&rsquo;s "
     "<b>target (mean) encoding</b>, which replaces each category with a smoothed estimate of the outcome "
     "rate for that category and is well suited to high-cardinality fields. Contact columns such as phone "
     "and e-mail are turned into <i>presence</i> and <i>validity</i> flags. Free-text columns are encoded "
     "with a pre-trained <b>sentence-transformer</b> model, <i>all-MiniLM-L6-v2</i>, which maps each text "
     "to a 384-dimensional embedding vector that captures its meaning; these vectors become additional "
     "numeric features. In this project 54 engineered features were produced from the raw columns.")
para("3.3 Models Compared", h3)
para("Four classifiers are trained and compared: <b>Logistic Regression</b> (a linear, highly "
     "interpretable model), a <b>Decision Tree</b>, a <b>Random Forest</b> (an ensemble of trees), and "
     "<b>XGBoost</b> (gradient-boosted trees). A <b>Dummy baseline</b> that always predicts the majority "
     "class is included as a no-skill reference, so that real skill can be distinguished from the effect "
     "of class imbalance.")
para("3.4 Honest Evaluation, Calibration and SHAP", h3)
para("Because conversion is rare, <b>accuracy alone is misleading</b> &mdash; a model that predicts "
     "&ldquo;no&rdquo; for everyone can still look accurate. LeadIQ therefore relies on <b>ROC-AUC</b> "
     "and, especially, <b>PR-AUC (average precision)</b>, together with precision, recall, F1 and "
     "<i>precision in the top&nbsp;10%</i>. Models are selected by training-only cross-validation and "
     "reported on a reserved test set with 95% bootstrap confidence intervals. Predicted probabilities "
     "are <b>calibrated</b> (isotonic regression) so that a score of, say, 0.3 really means about a 30% "
     "chance. Finally, <b>SHAP</b> (SHapley Additive exPlanations) assigns to every feature its "
     "contribution to a specific prediction, providing the &ldquo;why&rdquo; behind each score.")
para("3.5 Web Application Stack", h3)
para("The frontend is a single-page <b>React</b> application styled with Tailwind CSS, using Recharts for "
     "interactive charts and Framer Motion for animation. The backend is a <b>FastAPI</b> service that "
     "runs the machine-learning pipeline with scikit-learn, XGBoost, SHAP and sentence-transformers, and "
     "stores project metadata in <b>MongoDB</b>. A light theme and a nine-step guided journey make the "
     "workflow easy to follow.")
para("3.6 Summary of the Technology Stack", h3)
para("Table&nbsp;3.1 lists the main components with the versions used in the project.")
table([
    ["Component", "Technology", "Purpose"],
    ["Frontend framework", "React 18, React Router", "Single-page nine-step wizard UI"],
    ["Styling &amp; charts", "Tailwind CSS, Recharts, Framer Motion", "Layout, interactive charts, animation"],
    ["Web framework", "FastAPI (Python)", "REST API for the ML pipeline"],
    ["ML &amp; data", "scikit-learn 1.6.1, pandas 3.0.6, NumPy 2.2.6", "Pipelines, encoders, models, data handling"],
    ["Gradient boosting", "xgboost-cpu 3.0.4", "XGBoost classifier"],
    ["Explainability", "shap 0.48.0", "SHAP feature contributions"],
    ["Text embeddings", "sentence-transformers 3.4.1, transformers 4.49.0", "all-MiniLM-L6-v2 embeddings of free text"],
    ["Persistence", "MongoDB, joblib 1.4.2", "Project metadata and trained-model artefacts"],
    ["Export", "openpyxl 3.1.5, reportlab", "Excel workbooks and PDF reports"],
], [40, 60, 55], caption="Table 3.1: Technology stack of LeadIQ")
story.append(PageBreak())

# ============================================================= 4. SYSTEM REQUIREMENTS
para("4. System Requirements and Setup", h1)
para("4.1 Hardware Requirements", h3)
para("All computation runs locally on CPU; no GPU is required. The following are recommended:")
bullets([
    "A computer or server able to run Python&nbsp;3.11+, for example a dual- or quad-core processor with "
    "4&ndash;8&nbsp;GB of RAM (the text-embedding model is the most memory-intensive stage).",
    "Free disk space for the application, the uploaded dataset and the cached trained models.",
    "A current web browser (Chrome, Edge, Firefox or Safari) to use the interface.",
])
para("4.2 Software Requirements", h3)
table([
    ["Item", "Requirement"],
    ["Operating system", "Windows, Linux or macOS"],
    ["Language runtime", "Python 3.11 or newer, with pip"],
    ["Node.js / Yarn", "To build and run the React frontend"],
    ["Database", "MongoDB (local or hosted)"],
    ["Web browser", "A current version of Chrome, Edge, Firefox or Safari"],
], [50, 105], caption="Table 4.1: Software requirements")
para("4.3 Configuration Parameters", h3)
para("All settings are read from environment variables (a <font face='Courier'>.env</font> file). "
     "Table&nbsp;4.2 lists the key variables.")
table([
    ["Variable", "Location", "Meaning"],
    ["MONGO_URL", "backend/.env", "MongoDB connection string"],
    ["DB_NAME", "backend/.env", "Database name used by the backend"],
    ["REACT_APP_BACKEND_URL", "frontend/.env", "Base URL the frontend calls for the API"],
], [55, 45, 55], caption="Table 4.2: Configuration parameters")
para("4.4 Installation and Execution", h3)
para("The application is installed and started with the following steps:")
story.append(Paragraph(
    "# Backend<br/>"
    "cd backend<br/>"
    "pip install -r requirements.txt<br/>"
    "uvicorn server:app --host 0.0.0.0 --port 8001<br/><br/>"
    "# Frontend<br/>"
    "cd frontend<br/>"
    "yarn install<br/>"
    "yarn start", code))
para("The backend exposes its REST API under the <font face='Courier'>/api</font> prefix; the React "
     "frontend talks to it through the <font face='Courier'>REACT_APP_BACKEND_URL</font> variable.")
story.append(PageBreak())

# ============================================================= 5. SYSTEM DESIGN
para("5. System Design", h1)
para("5.1 Overall Workflow", h3)
para("Figure&nbsp;5.1 shows the journey of a dataset from upload to a ranked, explainable contact list. "
     "The nine steps map directly onto the five stages of the KDD process.")
framed_pic("ss_landing.png", 165, "Figure 5.1: The nine-step LeadIQ journey mapped onto the KDD process (landing page).")
para("Table&nbsp;5.1 maps each step to its KDD stage and to the backend module that performs the work.")
table([
    ["Step", "Stage (KDD)", "What happens", "Module"],
    ["1. Upload &amp; select", "Selection", "Parse file; choose target and positive class", "dataio.py"],
    ["2. Clean", "Pre-processing", "Duplicates, missing values, type fixes; quality report", "cleaning.py"],
    ["3. Explore", "Pre-processing", "Outcome rates and patterns across columns", "inspections.py"],
    ["4. Features", "Transformation", "Scaling, target encoding, flags, text embeddings", "features.py"],
    ["5. Train", "Data mining", "Cross-validation, model comparison, calibration", "training.py"],
    ["6. Evaluate", "Evaluation", "Ten-tab metrics, curves, threshold explorer", "evaluation.py"],
    ["7. Rank", "Interpretation", "Calibrated probabilities, percentile scores, tiers", "scoring.py"],
    ["8. Explain", "Interpretation", "Per-lead SHAP contributions", "explanations.py"],
    ["9. Export", "Knowledge use", "CSV, Excel and PDF downloads", "exports.py"],
], [34, 28, 62, 31], caption="Table 5.1: Workflow steps, KDD stages and the modules that implement them", font=9)
para("5.2 System Architecture", h3)
para("LeadIQ follows a layered architecture. The React browser application talks to a FastAPI backend "
     "over HTTP and JSON. The route handlers in <font face='Courier'>server.py</font> are thin: they "
     "validate requests, enforce a per-project lock so that a running training job cannot be disturbed, "
     "and call the service modules where the real work happens. The service layer uses scikit-learn, "
     "XGBoost, SHAP and sentence-transformers, and persists project state in MongoDB and trained "
     "artefacts on disk through joblib. Each project is identified by an id and a secret token and "
     "expires automatically after 24&nbsp;hours.")
para("5.3 Project Structure", h3)
table([
    ["Path", "Contents"],
    ["backend/server.py", "FastAPI application and all REST endpoints"],
    ["backend/dataio.py", "File parsing, sheet handling, target and leakage detection"],
    ["backend/cleaning.py", "Data cleaning and the quality report"],
    ["backend/inspections.py", "Exploratory analysis (outcome rates, profiles)"],
    ["backend/features.py", "Feature engineering, encoders and text embeddings"],
    ["backend/training.py", "Cross-validation, model comparison and calibration"],
    ["backend/evaluation.py", "Metrics, curves and threshold recalculation"],
    ["backend/explanations.py", "SHAP and permutation importance"],
    ["backend/scoring.py", "Probability scoring, percentile scores and tiers"],
    ["backend/exports.py", "CSV, Excel and PDF exports"],
    ["frontend/src/components/", "Landing page and the nine wizard steps"],
], [55, 100], caption="Table 5.2: Folder and file structure of the repository")
para("5.4 Data Model", h3)
para("Project state is stored in MongoDB. Two main collections are used, summarised in Table&nbsp;5.3.")
table([
    ["Collection", "Key fields"],
    ["projects", "id, token, created_at, expires_at, name, step, completed, target, positive, settings"],
    ["metrics", "project_id, best_model, split sizes, prevalence, roc_auc, pr_auc, calibration, threshold"],
], [35, 120], caption="Table 5.3: Main stored collections")
para("5.5 REST API Design", h3)
para("All endpoints are served under the <font face='Courier'>/api</font> prefix. Table&nbsp;5.4 lists "
     "the most important ones.")
table([
    ["Method", "Endpoint", "Description"],
    ["POST", "/api/projects", "Create a new project and return its token"],
    ["GET", "/api/projects/{pid}", "Return the project state and current step"],
    ["POST", "/api/projects/{pid}/upload", "Upload a dataset file"],
    ["POST", "/api/projects/{pid}/configure", "Set the target column and positive class"],
    ["POST", "/api/projects/{pid}/clean", "Run data cleaning"],
    ["GET", "/api/projects/{pid}/eda", "Return exploratory analysis"],
    ["POST", "/api/projects/{pid}/features", "Build the feature matrix"],
    ["POST", "/api/projects/{pid}/train", "Train and compare the models"],
    ["GET", "/api/projects/{pid}/metrics", "Return evaluation metrics and curves"],
    ["POST", "/api/projects/{pid}/threshold", "Recompute metrics at a new threshold"],
    ["GET", "/api/projects/{pid}/importance", "Return SHAP and permutation importance"],
    ["POST", "/api/projects/{pid}/score", "Score and rank the leads"],
    ["GET", "/api/projects/{pid}/explain/{lead_id}", "Explain a single lead with SHAP"],
    ["GET", "/api/projects/{pid}/export", "Download ranked leads / report"],
], [22, 78, 55], caption="Table 5.4: Key REST endpoints of LeadIQ", font=8.8)
para("5.6 User Interface Design", h3)
para("The interface is a single guided journey of nine steps shown in a stepper, with a light theme, a "
     "metrics guide, and clear loading, recalculating, unavailable and invalidated states. Changing a "
     "model-affecting choice (such as the target) invalidates the downstream steps so that stale results "
     "can never be mistaken for current ones. Table&nbsp;5.5 lists the steps.")
table([
    ["Step", "Purpose", "Main elements"],
    ["Upload &amp; select", "Bring in the data", "File drop zone, sample data, target and positive-class pickers"],
    ["Clean", "Fix data problems", "Quality report, cleaning actions"],
    ["Explore", "Understand patterns", "Outcome-rate charts by category, city, state"],
    ["Features", "Build the matrix", "Feature groups, embedding status"],
    ["Train", "Compare models", "Quick / thorough choice, training status and logs"],
    ["Evaluate", "Judge performance", "Ten tabs, model selector, threshold explorer"],
    ["Rank", "Prioritize leads", "Score cards, distribution chart, ranked table"],
    ["Explain", "See the why", "Lead picker, score, SHAP contribution chart"],
    ["Export", "Keep the evidence", "CSV, Excel and PDF downloads"],
], [30, 40, 85], caption="Table 5.5: Steps of the user interface", font=9)
story.append(PageBreak())

# ============================================================= 6. IMPLEMENTATION
para("6. Implementation and Working of the System", h1)
para("This chapter follows the order in which a dataset travels through the LeadIQ pipeline. For this "
     "report the system was run end to end on a realistic synthetic dataset of 6,000 business leads "
     "(<font face='Courier'>synthetic-business-leads.csv</font>) whose target column is "
     "<b>Converted</b> with positive class <b>yes</b>.")
para("6.1 Upload, Parsing and Selection", h3)
para("A dataset is uploaded with <font face='Courier'>POST /api/projects/{pid}/upload</font>. The "
     "<font face='Courier'>dataio</font> module parses CSV, TSV, TXT and Excel files, detects the "
     "delimiter and sheet, infers each column&rsquo;s role and type, and previews the data. The user "
     "then confirms the target column and the positive outcome. The system automatically checks for "
     "<b>leakage</b> &mdash; columns that would not be known at prediction time &mdash; and excludes "
     "them. In this run 6,000 labelled rows were loaded with no leakage detected.")
para("6.2 Data Cleaning", h3)
para("The <font face='Courier'>cleaning</font> module removes duplicate rows, reconciles inconsistent "
     "types, and handles missing values, producing a transparent quality report that records exactly "
     "what was changed. Rows with a missing label are excluded from training. Duplicate removal is done "
     "in a way that preserves the original train/test membership, so the reserved test set stays "
     "untouched.")
para("6.3 Exploratory Analysis", h3)
para("Step&nbsp;3 summarises the cleaned data: the overall positive-outcome rate and the outcome rate "
     "broken down by each categorical column. For this dataset 9.6% of the cleaned, labelled records "
     "have the positive outcome &mdash; a strong class imbalance of about 9.4&nbsp;:&nbsp;1 &mdash; which "
     "is why ranking metrics, not raw accuracy, drive the rest of the pipeline.")
pic("chart_imbalance.png", 110, "Figure 6.1: The target is strongly imbalanced &mdash; only 578 of 6,000 records converted.")
para("6.4 Feature Engineering", h3)
para("The <font face='Courier'>features</font> module turns the cleaned table into a purely numeric "
     "matrix inside a scikit-learn pipeline, so that exactly the same transformation is applied to "
     "training, test and new data:")
bullets([
    "<b>Numeric</b> columns (Rating, Review Count, Business Age, &hellip;) are imputed and scaled.",
    "<b>Categorical</b> columns (Category, City, State, Lead Source) are converted with smoothed "
    "<b>target encoding</b>.",
    "<b>Contact</b> columns (Website, Phone, Email, Social Media) become <i>presence</i> and "
    "<i>validity</i> flags &mdash; for example whether a phone number is valid.",
    "The free-text <b>Description</b> column is embedded with the pre-trained "
    "<i>all-MiniLM-L6-v2</i> sentence-transformer into a 384-dimensional vector.",
])
para("Together these produced <b>54 engineered features</b>. The embedding stage, which runs on CPU, "
     "took about 69&nbsp;seconds for this dataset. A stratified 80/20 split reserves an untouched test "
     "set before any model sees the data.")
para("6.5 Model Training and Selection", h3)
para("The <font face='Courier'>training</font> module runs <b>training-only cross-validation</b> to "
     "compare the four classifiers against the dummy baseline. Each model&rsquo;s hyper-parameters are "
     "tuned over a small search (8 candidates in the &ldquo;quick&rdquo; mode used here) using k-fold "
     "cross-validation <i>within the training set only</i>. The winner is chosen by <b>cross-validated "
     "average precision (PR-AUC)</b> &mdash; never by test-set performance &mdash; after which it is "
     "refit and its probabilities are <b>calibrated with isotonic regression</b>. For this run "
     "<b>Logistic Regression</b> (C&nbsp;=&nbsp;0.01) was selected, and the whole process took about "
     "84&nbsp;seconds.")
para("6.6 Evaluation and the Threshold Explorer", h3)
para("The reserved test set is scored once, at the very end, to give an honest estimate of real-world "
     "performance. The <font face='Courier'>evaluation</font> module computes accuracy, precision, "
     "recall, F1, ROC-AUC, PR-AUC and precision in the top&nbsp;10%, together with ROC, "
     "precision&ndash;recall, lift, cumulative-gain, learning and calibration curves, and 95% bootstrap "
     "confidence intervals. An <b>operating threshold</b> (0.114 here, chosen out-of-fold to suit the "
     "imbalance) converts probabilities into yes/no decisions. A <b>threshold explorer</b> lets the user "
     "move the threshold and immediately see the confusion matrix and threshold-dependent metrics "
     "recompute from the saved probabilities &mdash; without changing the probabilities, the ranking or "
     "the saved operating point.")
para("6.7 Scoring and Ranking", h3)
para("The <font face='Courier'>scoring</font> module applies the calibrated winner to produce, for each "
     "lead, a calibrated probability and a <b>percentile score from 0 to 100</b> (a rank relative to "
     "the training distribution, explicitly not a probability). Scores are grouped into four priority "
     "tiers &mdash; Very&nbsp;High, High, Medium and Low &mdash; and the leads are placed in a "
     "deterministic contact order. For the 1,167 held-out records, 136 fell into Very&nbsp;High, 257 "
     "into High, 325 into Medium and 449 into Low.")
para("6.8 Explanation with SHAP", h3)
para("For any chosen lead the <font face='Courier'>explanations</font> module computes <b>SHAP</b> "
     "values, showing how much each feature pushed that particular prediction up or down. The interface "
     "states clearly that SHAP explains the underlying <i>uncalibrated</i> estimator, not the calibrated "
     "probability or the percentile score, and that contributions are associations, not causation. Global "
     "importance is also provided (SHAP and permutation), with sampling limitations disclosed.")
para("6.9 Export", h3)
para("Finally the <font face='Courier'>exports</font> module writes the ranked leads to CSV, a formatted "
     "Excel workbook (with a frozen header and safe cells) and a PDF summary containing the key metrics, "
     "the model comparison and the top-ranked records, so that the evidence travels with the result.")
story.append(PageBreak())

# ============================================================= 7. RESULTS
para("7. Results and Output Screens", h1)
para("This chapter presents the real output of the deployed LeadIQ application, in the order a user "
     "experiences it, together with the charts produced from the actual run described in Chapter&nbsp;6.")
para("7.1 Upload &amp; Select", h3)
framed_pic("ss_step1_selection.png", 165, "Figure 7.1: Step 1 &mdash; uploading a dataset and selecting the target and positive class.")
para("7.2 Explore the Data", h3)
framed_pic("ss_step3_explore.png", 165, "Figure 7.2: Step 3 &mdash; exploratory analysis of outcome rates across categories, cities and states.")
story.append(PageBreak())
para("7.3 Train the Models", h3)
framed_pic("ss_step5_train.png", 165, "Figure 7.3: Step 5 &mdash; training complete; Logistic Regression selected by cross-validated PR-AUC in 84&nbsp;s.")
para("7.4 Model Comparison", h3)
para("All five candidates were measured on the same untouched test set. Table&nbsp;7.1 and "
     "Figure&nbsp;7.4 report the results; the Dummy baseline shows why accuracy alone is misleading "
     "(it reaches 0.901 accuracy with zero recall).")
table([
    ["Model", "Acc.", "Prec.", "Recall", "F1", "ROC-AUC", "PR-AUC", "P@10%", "CV&nbsp;PR-AUC"],
    [Paragraph("<b>Logistic Regression (winner)</b>", cell), "0.743", "0.208", "0.574", "0.306", "0.711", "0.271", "0.291", "0.187"],
    ["Decision Tree", "0.683", "0.169", "0.565", "0.260", "0.647", "0.156", "0.214", "0.148"],
    ["Random Forest", "0.862", "0.298", "0.296", "0.297", "0.699", "0.232", "0.291", "0.154"],
    ["XGBoost", "0.681", "0.171", "0.583", "0.265", "0.717", "0.271", "0.282", "0.176"],
    ["Dummy baseline", "0.901", "0.000", "0.000", "0.000", "0.500", "0.099", "0.128", "0.096"],
], [40, 14, 15, 16, 14, 20, 19, 17, 20, ][:9], caption="Table 7.1: Test-set results for all models (winner selected by CV PR-AUC).", font=8.6)
pic("chart_model_compare.png", 160, "Figure 7.4: ROC-AUC and PR-AUC for each model on the untouched test set.")
framed_pic("ss_step6_compare.png", 168, "Figure 7.5: Step 6 &mdash; the Model Comparison tab in the application.")
story.append(PageBreak())
para("7.5 Evaluation Metrics of the Selected Model", h3)
para("Table&nbsp;7.2 summarises the winning model&rsquo;s test-set metrics with 95% bootstrap confidence "
     "intervals (1,000 resamples), and Figure&nbsp;7.6 shows the Overview tab.")
table([
    ["Metric", "Value", "95% confidence interval"],
    ["Accuracy", "0.743", "&ndash;"],
    ["Precision", "0.208", "&ndash;"],
    ["Recall", "0.574", "&ndash;"],
    ["F1 score", "0.306", "0.252 &ndash; 0.363"],
    ["ROC-AUC", "0.711", "0.660 &ndash; 0.765"],
    ["PR-AUC (avg precision)", "0.271", "0.208 &ndash; 0.354"],
    ["Precision @ top 10%", "0.291", "2.95&times; lift over random"],
    ["Operating threshold", "0.114", "chosen out-of-fold"],
], [55, 35, 65], caption="Table 7.2: Test-set metrics of the selected Logistic Regression model.", font=9)
framed_pic("ss_step6_overview.png", 168, "Figure 7.6: Step 6 &mdash; the Overview tab with the seven headline metrics and dataset context.")
story.append(PageBreak())
para("7.6 Curves and Diagnostics", h3)
para("The figures below are drawn from the actual saved predictions of the selected model.")
# two-up: ROC and PR
def two_up(n1, n2, cap1, cap2, w=78):
    p1 = os.path.join(AST, n1); p2 = os.path.join(AST, n2)
    def mkimg(p):
        iw, ih = PILImage.open(p).size
        ww = w * mm; hh = ww * ih / iw
        return Image(p, width=ww, height=hh)
    t = Table([[mkimg(p1), mkimg(p2)],
               [Paragraph(cap1, cap), Paragraph(cap2, cap)]], colWidths=[w * mm, w * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    t.hAlign = "CENTER"
    story.append(t)
two_up("chart_roc.png", "chart_pr.png",
       "Figure 7.7: ROC curve (AUC = 0.711).",
       "Figure 7.8: Precision&ndash;Recall curve (AP = 0.271).")
two_up("chart_confusion.png", "chart_calibration.png",
       "Figure 7.9: Confusion matrix at the operating threshold.",
       "Figure 7.10: Calibration reliability (before / after isotonic).")
para("The confusion matrix at the 0.114 threshold yields 66 true positives and 801 true negatives, "
     "against 251 false positives and 49 false negatives &mdash; a recall-leaning operating point that "
     "deliberately surfaces more candidates, as is sensible for prioritizing outreach.")
story.append(PageBreak())
two_up("chart_learning.png", "chart_shap.png",
       "Figure 7.11: Learning curve of the selected model.",
       "Figure 7.12: Global feature importance (mean |SHAP|).")
para("7.7 Lift and Cumulative Gain", h3)
para("Lift and cumulative-gain curves show the business value of the ranking: by contacting only the "
     "top-scored leads, far more conversions are captured than by contacting at random.")
pic("chart_lift_gain.png", 168, "Figure 7.13: Lift (left) and cumulative gain (right) for the LeadIQ ranking.")
framed_pic("ss_step6_lift.png", 165, "Figure 7.14: Step 6 &mdash; the Lift &amp; Gain tab in the application.")
story.append(PageBreak())
para("7.8 Score &amp; Rank Leads", h3)
para("The selected model scored the 1,167 held-out leads into percentile scores and four priority tiers. "
     "Figure&nbsp;7.15 shows the distribution and Figure&nbsp;7.16 the ranked list.")
two_up("chart_scores.png", "chart_tiers.png",
       "Figure 7.15: Distribution of 0&ndash;100 lead scores by tier.",
       "Figure 7.16: Share of leads in each priority tier.")
framed_pic("ss_step7_ranking.png", 165, "Figure 7.17: Step 7 &mdash; score cards and priority distribution.")
framed_pic("ss_step7_table.png", 165, "Figure 7.18: Step 7 &mdash; the ranked lead table in deterministic contact order.")
story.append(PageBreak())
para("7.9 Explain a Lead", h3)
para("For the top-ranked lead (score 100, calibrated probability 66.7%, priority Very&nbsp;High), SHAP "
     "shows which features pushed the prediction up or down. The interface is explicit that SHAP "
     "explains the uncalibrated estimator, not the percentile score, and that the contributions are "
     "associations rather than causation.")
framed_pic("ss_step8_explain.png", 165, "Figure 7.19: Step 8 &mdash; a per-lead SHAP explanation.")
framed_pic("ss_step8_shap.png", 165, "Figure 7.20: Step 8 &mdash; the SHAP contribution chart for the selected lead.")
story.append(PageBreak())
para("7.10 Export &amp; Summary", h3)
framed_pic("ss_step9_export.png", 165, "Figure 7.21: Step 9 &mdash; CSV, Excel and PDF export of the ranked leads and summary.")
para("7.11 Discussion", h3)
para("The results show both the strengths and the honesty of the design. On a hard, strongly imbalanced "
     "problem the selected model reaches a PR-AUC of 0.271 against a no-information baseline of 0.099 "
     "&mdash; roughly a <b>17 percentage-point</b> improvement &mdash; and a ROC-AUC of 0.711, with "
     "about <b>2.95&times; lift</b> in the top decile. In practice this means a team working the "
     "top-scored leads would reach far more converters than one working the list at random. Equally "
     "important, every number is measured on data the model never saw, reported with a confidence "
     "interval, and backed by a SHAP explanation, so the user can see not only the ranking but the "
     "reasoning behind it.")
story.append(PageBreak())

# ============================================================= 8. ADV / LIM / FUTURE
para("8. Advantages, Limitations and Future Scope", h1)
para("8.1 Advantages", h3)
bullets([
    "<b>Dataset-agnostic.</b> Any labelled tabular file can be scored; the system is not tied to a fixed "
    "lead schema and builds its features automatically.",
    "<b>Honest evaluation.</b> Models are selected by training-only cross-validation and reported on an "
    "untouched test set with bootstrap confidence intervals, so the numbers are trustworthy.",
    "<b>Imbalance-aware.</b> Ranking metrics (PR-AUC, lift, precision in the top&nbsp;10%) and a tuned "
    "operating threshold are used instead of misleading raw accuracy.",
    "<b>Calibrated probabilities.</b> Isotonic calibration makes a score of 0.3 really mean about a 30% "
    "chance, which matters when the output drives real decisions.",
    "<b>Explainable.</b> Every prediction comes with real SHAP contributions and a global importance "
    "view, with limitations disclosed rather than hidden.",
    "<b>Guided and safe.</b> The nine-step KDD workflow invalidates downstream results when an upstream "
    "choice changes, so stale numbers can never be mistaken for current ones.",
])
para("8.2 Limitations", h3)
bullets([
    "The dataset used in this report is <b>synthetic</b>; it demonstrates the full pipeline but not "
    "real-world predictive power. On this hard problem the absolute precision is modest.",
    "Text embeddings run on CPU with the English-first <i>all-MiniLM-L6-v2</i> model; long or non-English "
    "text is truncated and may be embedded less reliably.",
    "SHAP and permutation importance are computed on a sample for speed, so very small differences in "
    "importance should not be over-interpreted.",
    "Scores reflect <b>association</b> learned from historical data, not causation, and depend on the "
    "quality and representativeness of the uploaded data.",
])
para("8.3 Future Scope", h3)
bullets([
    "Save and compare separate historical runs side by side as an explicit history feature.",
    "Monitor predictions against later real-world outcomes and detect performance drift over time.",
    "Add GPU-accelerated and multilingual text embeddings for larger, non-English datasets.",
    "Support multi-class and regression targets in addition to binary classification.",
    "Add role-based accounts and team sharing for collaborative outreach.",
])
story.append(PageBreak())

# ============================================================= 9. CONCLUSION
para("9. Conclusion", h1)
para("LeadIQ shows that a complete, practical and <b>trustworthy</b> lead-scoring tool can be built by "
     "combining classical machine learning with careful engineering around the models. A user uploads a "
     "spreadsheet; the system cleans it, builds 54 features including sentence-transformer text "
     "embeddings, compares four classifiers against a no-skill baseline with training-only "
     "cross-validation, calibrates the winner, chooses an operating threshold, and returns a ranked, "
     "tiered contact list in which every score is explained with real SHAP values.")
para("The main lesson of the project is the same one that runs through the whole KDD process: model "
     "output must not be trusted blindly. Reserving an untouched test set, reporting confidence "
     "intervals, preferring ranking metrics over accuracy on an imbalanced problem, calibrating "
     "probabilities, and explaining every prediction are what turn a bare classifier into a dependable "
     "decision aid. On a deliberately hard synthetic dataset the selected model still delivered about a "
     "17 percentage-point improvement in PR-AUC over the baseline and roughly 2.95&times; lift in the "
     "top decile &mdash; while never hiding its uncertainty. The result is a system that answers not "
     "only &ldquo;who is the next best lead?&rdquo; but also &ldquo;why?&rdquo;.")
story.append(PageBreak())

# ============================================================= 10. REFERENCES
para("10. References", h1)
refs = [
    "U. Fayyad, G. Piatetsky-Shapiro and P. Smyth, &ldquo;From Data Mining to Knowledge Discovery in "
    "Databases,&rdquo; <i>AI Magazine</i>, vol.&nbsp;17, no.&nbsp;3, pp.&nbsp;37&ndash;54, 1996.",
    "F. Pedregosa et al., &ldquo;Scikit-learn: Machine Learning in Python,&rdquo; <i>Journal of Machine "
    "Learning Research</i>, vol.&nbsp;12, pp.&nbsp;2825&ndash;2830, 2011.",
    "scikit-learn developers, &ldquo;TargetEncoder &mdash; target (mean) encoding of categorical "
    "features,&rdquo; scikit-learn User Guide, https://scikit-learn.org/stable/modules/preprocessing_targetencoder.html.",
    "T. Chen and C. Guestrin, &ldquo;XGBoost: A Scalable Tree Boosting System,&rdquo; in <i>Proc. 22nd "
    "ACM SIGKDD</i>, 2016, pp.&nbsp;785&ndash;794.",
    "S. M. Lundberg and S.-I. Lee, &ldquo;A Unified Approach to Interpreting Model Predictions,&rdquo; in "
    "<i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2017.",
    "N. Reimers and I. Gurevych, &ldquo;Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,&rdquo; "
    "in <i>Proc. EMNLP</i>, 2019. (Model: all-MiniLM-L6-v2.)",
    "B. Zadrozny and C. Elkan, &ldquo;Transforming Classifier Scores into Accurate Multiclass Probability "
    "Estimates,&rdquo; in <i>Proc. 8th ACM SIGKDD</i>, 2002. (Isotonic calibration.)",
    "T. Saito and M. Rehmsmeier, &ldquo;The Precision-Recall Plot Is More Informative than the ROC Plot "
    "When Evaluating Binary Classifiers on Imbalanced Datasets,&rdquo; <i>PLoS ONE</i>, 10(3), 2015.",
    "S. Ram&iacute;rez, &ldquo;FastAPI framework,&rdquo; https://fastapi.tiangolo.com.",
    "Meta Open Source, &ldquo;React &mdash; A JavaScript library for building user interfaces,&rdquo; "
    "https://react.dev.",
]
story.append(ListFlowable(
    [ListItem(Paragraph(r, mk("ref", fontName="Times-Roman", fontSize=10.5, leading=14,
                              alignment=TA_JUSTIFY, textColor=INK)), leftIndent=6) for r in refs],
    bulletType="1", bulletFormat="[%s]", leftIndent=22, spaceAfter=8))

# ============================================================= page decoration
def decorate(canvas, doc):
    canvas.saveState()
    page = doc.page
    if page == 1:
        # outer border box on the cover
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(1.4)
        m = 12 * mm
        canvas.rect(m, m, PAGE_W - 2 * m, PAGE_H - 2 * m)
        canvas.setLineWidth(0.6)
        canvas.rect(m + 2, m + 2, PAGE_W - 2 * m - 4, PAGE_H - 2 * m - 4)
    # running header
    canvas.setFont("Times-Italic", 8.5)
    canvas.setFillColor(GREY)
    canvas.drawString(MARGIN, PAGE_H - 13 * mm, HEAD_L)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 13 * mm, HEAD_R)
    canvas.setStrokeColor(LIGHT)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, PAGE_H - 14.5 * mm, PAGE_W - MARGIN, PAGE_H - 14.5 * mm)
    # footer page number
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(PAGE_W / 2, 10 * mm, str(page))
    canvas.restoreState()

doc = BaseDocTemplate(OUTPUT, pagesize=A4,
                      leftMargin=MARGIN, rightMargin=MARGIN,
                      topMargin=18 * mm, bottomMargin=16 * mm,
                      title="LeadIQ Project Report", author="Darshil Savaliya, Harshil Patel, Aranav Mandani")
frame = Frame(MARGIN, 16 * mm, PAGE_W - 2 * MARGIN, PAGE_H - 34 * mm, id="main")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])
doc.build(story)
print("PDF written to", OUTPUT)
import os as _os
print("Size:", round(_os.path.getsize(OUTPUT) / 1024, 1), "KB")
