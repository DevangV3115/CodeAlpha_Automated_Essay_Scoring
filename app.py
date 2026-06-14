import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pypdf import PdfReader
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Import pipeline components
from src.predict import score_essay, pipeline
from src.train import train_pipeline

# Page Configuration
st.set_page_config(
    page_title="Automated Essay Scoring (AES) Dashboard",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern dark glassmorphism look
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0A0E17;
        color: #E2E8F0;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #00F2FE !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #161B25;
        border-right: 1px solid #1E293B;
    }
    
    /* Custom Cards */
    .metric-card {
        background: rgba(22, 27, 37, 0.7);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .metric-val-high {
        font-size: 48px;
        font-weight: bold;
        color: #10B981; /* Green */
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }
    .metric-val-med {
        font-size: 48px;
        font-weight: bold;
        color: #F59E0B; /* Orange */
        text-shadow: 0 0 10px rgba(245, 158, 11, 0.2);
    }
    .metric-val-low {
        font-size: 48px;
        font-weight: bold;
        color: #EF4444; /* Red */
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.2);
    }
    
    .metric-label {
        font-size: 14px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 5px;
    }
    
    /* Custom Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        color: #0A0E17;
        font-weight: bold;
        border: none;
        padding: 10px 28px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 242, 254, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# FILE READERS HELPERS
# ==========================================
def extract_text(file) -> str:
    """Extracts text from uploaded txt, docx, or pdf files."""
    filename = file.name
    try:
        if filename.endswith(".txt"):
            return file.read().decode("utf-8")
        elif filename.endswith(".docx"):
            doc = Document(io.BytesIO(file.read()))
            return "\n".join([para.text for para in doc.paragraphs])
        elif filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file.read()))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        st.error(f"Error reading file {filename}: {e}")
    return ""


# ==========================================
# REPORT GENERATION HELPER
# ==========================================
def make_pdf_report(essay_text: str, score_data: dict) -> bytes:
    """Generates a styled downloadable PDF scoring report using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=45, leftMargin=45, topMargin=40, bottomMargin=40
    )
    story = []
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor('#00F2FE')
    c_dark = colors.HexColor('#0A0E17')
    c_gray = colors.HexColor('#64748B')
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'],
        fontSize=22, textColor=c_dark, spaceAfter=8, alignment=0
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontSize=10, textColor=c_gray, spaceAfter=20
    )
    header_style = ParagraphStyle(
        'SectionHeader', parent=styles['Heading2'],
        fontSize=14, textColor=c_dark, spaceBefore=12, spaceAfter=8, keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyTextCustom', parent=styles['Normal'],
        fontSize=10, spaceAfter=8, leading=14
    )
    
    # Title / Metadata
    story.append(Paragraph("AUTOMATED ESSAY SCORING REPORT", title_style))
    story.append(Paragraph("Generated offline using NLP & Machine Learning algorithms", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Score Summary Table
    final_score = score_data["final_score"]
    cat = score_data["category_scores"]
    
    data = [
        [Paragraph("<b>Category</b>", body_style), Paragraph("<b>Rating / 10.0</b>", body_style)],
        [Paragraph("<b>OVERALL SCORE</b>", body_style), Paragraph(f"<b>{final_score}</b>", body_style)],
        [Paragraph("Content Relevance", body_style), Paragraph(str(cat['content']), body_style)],
        [Paragraph("Grammar & Mechanics", body_style), Paragraph(str(cat['grammar']), body_style)],
        [Paragraph("Coherence & Structure", body_style), Paragraph(str(cat['coherence']), body_style)],
        [Paragraph("Vocabulary & Richness", body_style), Paragraph(str(cat['vocabulary']), body_style)]
    ]
    
    t = Table(data, colWidths=[200, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#F1F5F9')),
        ('BACKGROUND', (0,1), (1,1), colors.HexColor('#ECFDF5')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
    ]))
    
    story.append(Paragraph("Score Dashboard", header_style))
    story.append(t)
    story.append(Spacer(1, 15))
    
    # Strengths / Weaknesses / Suggestions
    story.append(Paragraph("Personalized Diagnostic Feedback", header_style))
    
    story.append(Paragraph("<b>Key Strengths:</b>", body_style))
    for s in score_data["feedback"]["strengths"]:
        story.append(Paragraph(f"• {s}", body_style))
        
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Areas for Improvement:</b>", body_style))
    for w in score_data["feedback"]["weaknesses"]:
        story.append(Paragraph(f"• {w}", body_style))
        
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Actionable Suggestions:</b>", body_style))
    for sug in score_data["feedback"]["suggestions"]:
        story.append(Paragraph(f"• {sug}", body_style))
        
    story.append(Spacer(1, 15))
    
    # Original essay (Truncated if too long)
    story.append(Paragraph("Submitted Essay Text", header_style))
    truncated_essay = essay_text if len(essay_text) < 1800 else essay_text[:1800] + "..."
    # Format newlines
    for para in truncated_essay.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.replace("\n", "<br/>"), body_style))
            
    doc.build(story)
    return buffer.getvalue()


# ==========================================
# STREAMLIT SIDEBAR
# ==========================================
st.sidebar.title("✍️ AES Engine Control")

# 1. Project Overview
with st.sidebar.expander("ℹ️ Project Info", expanded=True):
    st.markdown("""
    **Automated Essay Scoring (AES)** system powered by **NLP** and **supervised machine learning**.
    
    - **Offline Capabilities**: No paid external APIs are utilized.
    - **Evaluation Scope**: Assesses Content, Grammar, Coherence, and Vocabulary.
    - **Resource Limit**: Optimized for low RAM usage and rapid load speeds.
    """)

# 2. Dataset Information
with st.sidebar.expander("📊 Dataset Details"):
    st.markdown("""
    **Dataset**: ASAP Automated Student Assessment Prize (Kaggle)
    
    **Features**:
    - 8 distinct essay prompt groups.
    - Double-graded human scores for validation.
    
    **Structure**:
    - `essay_id`: Unique identifier
    - `essay_set`: Prompt group ID
    - `essay`: Raw text
    - `domain1_score`: Target grade
    """)

# 3. Model Information
with st.sidebar.expander("🤖 Trained Estimator"):
    if pipeline.fallback_mode:
        st.warning("⚠️ Traditional ML model weights not found. Using Rule-Based Heuristics engine.")
    else:
        st.success("✅ Supervised Random Forest pipeline is active.")
        
    st.markdown("""
    - **Traditional ML**: Random Forest Regressor & XGBoost (Optimized via CV).
    - **Deep Learning**: PyTorch LSTM & GRU architectures.
    - **Inference Speed**: ~40ms on standard CPU.
    """)

# 4. Settings & Training Panel
with st.sidebar.expander("⚙️ Controls & Training"):
    # One-click generator and trainer
    if st.button("Generate Synthetic Data & Train"):
        with st.spinner("Generating mock data, extracting features, and training..."):
            try:
                train_pipeline(generate_synthetic_flag=True)
                # Re-init pipeline
                pipeline.load_pipeline()
                st.success("Models trained successfully! Refreshed scoring pipeline.")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

# ==========================================
# MAIN PAGE
# ==========================================
st.title("Automated Essay Scoring System")
st.write("Submit your essay below manually, or upload a document to generate instantaneous feedback.")

# Core layout: two tabs (1: Scorer, 2: Dataset EDA Explorer)
tab_scorer, tab_eda = st.tabs(["✍️ Essay Evaluator", "📈 Dataset Analytics & EDA"])

with tab_scorer:
    col_input, col_meta = st.columns([2, 1])
    
    with col_meta:
        st.subheader("Scoring Metadata")
        prompt_text = st.text_area(
            "Target Prompt / Question Topic (Optional)",
            placeholder="e.g. Should computers replace teachers in the classroom?",
            help="Providing a prompt allows the engine to calculate thematic semantic similarity metrics."
        )
        
        st.markdown("### Document Upload")
        uploaded_file = st.file_uploader(
            "Upload TXT, DOCX, or PDF files",
            type=["txt", "docx", "pdf"]
        )

    with col_input:
        st.subheader("Essay Input")
        
        # Populate text area if document is uploaded
        initial_text = ""
        if uploaded_file is not None:
            initial_text = extract_text(uploaded_file)
            
        essay_text = st.text_area(
            "Write or Paste your essay here (minimum 30 words suggested):",
            value=initial_text,
            height=300,
            placeholder="Begin typing your essay..."
        )
        
        evaluate_btn = st.button("Evaluate Essay")

    # Scoring Logic Trigger
    if evaluate_btn:
        if len(essay_text.strip().split()) < 10:
            st.error("Please enter a valid essay containing at least 10 words.")
        else:
            with st.spinner("Analyzing essay structures, syntax, spelling, and semantic alignment..."):
                results = score_essay(essay_text, prompt_text if prompt_text.strip() else None)
                
                final_score = results["final_score"]
                subscores = results["category_scores"]
                feedback = results["feedback"]
                features = results["features"]
                
                st.success("Essay Evaluated Successfully!")
                
                # --- RESULTS DASHBOARD ---
                st.write("---")
                st.subheader("Evaluation Results Dashboard")
                
                col_score, col_radar = st.columns([1, 2])
                
                with col_score:
                    # Select color based on final score range
                    if final_score >= 7.5:
                        val_class = "metric-val-high"
                    elif final_score >= 5.0:
                        val_class = "metric-val-med"
                    else:
                        val_class = "metric-val-low"
                        
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="{val_class}">{final_score}</div>
                        <div class="metric-label">Overall Essay Score / 10.0</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Print subscores as individual gauges/progress bars
                    st.write("**Category Breakdown:**")
                    for cat_name, val in subscores.items():
                        st.write(f"{cat_name.capitalize()}: **{val}/10.0**")
                        st.progress(val / 10.0)

                with col_radar:
                    # Radar Chart using Plotly
                    categories = ['Content Relevance', 'Grammar & Mechanics', 'Coherence & Structure', 'Vocabulary Richness']
                    scores_list = [subscores['content'], subscores['grammar'], subscores['coherence'], subscores['vocabulary']]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=scores_list,
                        theta=categories,
                        fill='toself',
                        fillcolor='rgba(0, 242, 254, 0.2)',
                        line=dict(color='#00F2FE', width=2),
                        name='Essay Scores'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 10]),
                            bgcolor='rgba(22, 27, 37, 0.8)'
                        ),
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=30, b=30, l=40, r=40),
                        height=280
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # --- FEEDBACK PANEL ---
                st.subheader("Personalized Diagnostic Feedback")
                col_str, col_weak, col_sug = st.columns(3)
                
                with col_str:
                    st.info("🌟 **Key Strengths**")
                    for s in feedback["strengths"]:
                        st.write(f"- {s}")
                        
                with col_weak:
                    st.warning("⚠️ **Areas for Improvement**")
                    for w in feedback["weaknesses"]:
                        st.write(f"- {w}")
                        
                with col_sug:
                    st.success("💡 **Actionable Suggestions**")
                    for sug in feedback["suggestions"]:
                        st.write(f"- {sug}")
                        
                # --- ANALYSIS PANELS ---
                st.subheader("Deep-Dive Linguistic Analytics")
                tab_grammar, tab_vocab, tab_read, tab_stats = st.tabs([
                    "✏️ Grammar & Spelling", 
                    "📖 Vocabulary Analysis", 
                    "👁️ Readability Indices", 
                    "📊 General Statistics"
                ])
                
                with tab_grammar:
                    col_g1, col_g2 = st.columns(2)
                    col_g1.metric("Grammar Flaws Detected", int(features["grammar_errors"]))
                    col_g2.metric("Spelling Mistakes", int(features["spelling_errors"]))
                    st.info("Spelling accuracy is estimated using the TextBlob pattern-matching dictionary. Heuristics search for missing apostrophes, run-on descriptors, and passive constructs.")
                    
                with tab_vocab:
                    col_v1, col_v2, col_v3 = st.columns(3)
                    col_v1.metric("Unique Word Ratio (TTR)", f"{features['type_token_ratio']:.2f}")
                    col_v2.metric("Lexical Diversity (Guiraud)", f"{features['lexical_diversity']:.2f}")
                    col_v3.metric("Advanced Vocabulary Ratio", f"{features['advanced_vocab_ratio']*100:.1f}%")
                    st.write("A higher Type-Token Ratio indicates complex, descriptive phrasing and avoids word repetitions.")
                    
                with tab_read:
                    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                    col_r1.metric("Flesch Ease", f"{features['flesch_reading_ease']:.1f}")
                    col_r2.metric("Kincaid Grade", f"{features['flesch_kincaid_grade']:.1f}")
                    col_r3.metric("Gunning Fog", f"{features['gunning_fog']:.1f}")
                    col_r4.metric("Dale-Chall", f"{features['dale_chall']:.2f}")
                    st.write("Readability grade scales map the sentence structures and syllable densities to school levels (e.g. Grade 8 matches early high school writing).")
                    
                with tab_stats:
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    col_s1.metric("Word Count", int(features["word_count"]))
                    col_s2.metric("Character Count", int(features["char_count"]))
                    col_s3.metric("Paragraphs", int(features["paragraph_count"]))
                    col_s4.metric("Sentences", int(features["sentence_count"]))
                
                # --- PDF GENERATOR ---
                st.write("---")
                pdf_bytes = make_pdf_report(essay_text, {
                    "final_score": final_score,
                    "category_scores": subscores,
                    "feedback": feedback
                })
                
                st.download_button(
                    label="📥 Download PDF Evaluation Report",
                    data=pdf_bytes,
                    file_name="essay_evaluation_report.pdf",
                    mime="application/pdf"
                )

with tab_eda:
    st.subheader("Dataset Exploratory Data Analysis (EDA)")
    
    # Check if we have synthetic essays to show live metrics
    csv_path = "data/synthetic_essays.csv"
    if os.path.exists(csv_path):
        df_eda = pd.read_csv(csv_path)
        
        col_st1, col_st2, col_st3 = st.columns(3)
        col_st1.metric("Total Essays in Database", len(df_eda))
        col_st2.metric("Mean Essay Score", f"{df_eda['domain1_score'].mean():.2f}")
        col_st3.metric("Missing Value Entries", df_eda.isnull().sum().sum())
        
        st.write("### Essay Score Distribution")
        fig_dist = px.histogram(
            df_eda, x="domain1_score", 
            nbins=15, 
            labels={"domain1_score": "Score (1.0 - 10.0)"},
            color_discrete_sequence=["#00F2FE"]
        )
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0')
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.write("### Dataset Sample Breakdown")
        st.dataframe(df_eda[["essay_id", "essay_set", "essay", "domain1_score"]].head(10))
    else:
        st.info("Live dataset graphs are available once synthetic essay data is trained. Check 'Controls & Training' in the sidebar to create/train a local database.")
        st.markdown("""
        #### ASAP Dataset Statistics (Standard Baseline)
        - **Total Essays**: 12,976 records.
        - **Mean Score**: 61% (across heterogeneous scales).
        - **Essay Sets**: 8 prompts.
        - **Missing values**: 0.0% missing text cells.
        """)
