import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
from docx import Document
import google.generativeai as genai
import re
import difflib

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

def extract_text_from_pdf(file_obj):
    reader = PyPDF2.PdfReader(file_obj)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_obj):
    doc = Document(file_obj)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def compare_with_gemini(cds_text, child_text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""
You are an expert engine trained on global pharma document formats (CDS, USPI, SmPC, SwissPI, JPI, AUSPI, IPL). Your task: analyze a source (CDS) and a child (e.g., USPI) document, and report section-level differences.

TASK:
For each aligned section:
Analyze both sections fully.
Identify differences as:
Added (child-only)
Omitted (CDS-only)
Modified (content differs)
Summarize differences in bullet points.
Extract verbatim the differing text.

OUTPUT FORMAT:
Section: [Section Title]
Summary of Differences:
[Bullet point summary of each difference]
CDS-only (green):
"[Exact CDS text missing from child]"
Child-only (red):
"[Exact child text not in CDS]"
Modified (pink):
"[CDS text]" / "[Child text]"

If no differences, output:
Section: [Section Title]
No differences found.

CONSTRAINTS:
Ignore identical content and formatting differences.
Do not paraphrase or comment; quote differences verbatim.
No regulatory strategy or intent analysis.

CDS Section:
{cds_text}

Child Section:
{child_text}
"""
    response = model.generate_content(prompt)
    return response.text

def split_into_sections(text):
    # This regex matches headings like '1. ', '2.1 ', etc.
    pattern = r'(\d+(?:\.\d+)*\.\s+[^\n]+)'
    matches = list(re.finditer(pattern, text))
    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        title = match.group().strip()
        content = text[start+len(title):end].strip()
        sections.append({'title': title, 'content': content})
    return sections

def highlight_differences(cds_text, child_text):
    differ = difflib.Differ()
    diff = list(differ.compare(cds_text.split(), child_text.split()))
    cds_highlighted = []
    child_highlighted = []
    for word in diff:
        if word.startswith('- '):
            cds_highlighted.append(f'<span class="text-green-600">{word[2:]}</span>')
        elif word.startswith('+ '):
            child_highlighted.append(f'<span class="text-red-600">{word[2:]}</span>')
        elif word.startswith('  '):
            cds_highlighted.append(word[2:])
            child_highlighted.append(word[2:])
    return ' '.join(cds_highlighted), ' '.join(child_highlighted)

def compare_section_content(cds_content, child_content):
    matcher = difflib.SequenceMatcher(None, cds_content, child_content)
    cds_only = []
    child_only = []
    modified = []
    summary = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            cds_text = cds_content[i1:i2].strip()
            child_text = child_content[j1:j2].strip()
            if cds_text and child_text:
                modified.append({"cds": cds_text, "child": child_text})
                summary.append(f'Modified: "{cds_text}" → "{child_text}"')
        elif tag == 'delete':
            cds_text = cds_content[i1:i2].strip()
            if cds_text:
                cds_only.append(cds_text)
                summary.append(f'Omitted in child: "{cds_text}"')
        elif tag == 'insert':
            child_text = child_content[j1:j2].strip()
            if child_text:
                child_only.append(child_text)
                summary.append(f'Added in child: "{child_text}"')

    no_difference = not (cds_only or child_only or modified)
    return {
        "summary": summary,
        "cds_only": cds_only,
        "child_only": child_only,
        "modified": modified,
        "no_difference": no_difference
    }

def strict_sectionwise_comparison(cds_sections, child_sections):
    cds_dict = {s['title']: s['content'] for s in cds_sections}
    child_dict = {s['title']: s['content'] for s in child_sections}
    all_titles = sorted(set(cds_dict.keys()) | set(child_dict.keys()))
    results = []
    for title in all_titles:
        cds_content = cds_dict.get(title, '')
        child_content = child_dict.get(title, '')
        section_result = compare_section_content(cds_content, child_content)
        results.append({
            "section": title,
            **section_result
        })
    return results

def find_best_match(section, candidates):
    best_score = 0
    best_match = None
    for cand in candidates:
        score = difflib.SequenceMatcher(None, section['content'], cand['content']).ratio()
        if score > best_score:
            best_score = score
            best_match = cand
    return best_match, best_score

def summarize_section_with_gemini(cds_text, child_text, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""
Summarize the following two sections and their differences in 100 words or less:
CDS Section:
{cds_text}

Child Section:
{child_text}
"""
    response = model.generate_content(prompt)
    return response.text.strip()

def match_and_summarize_sections(cds_sections, child_sections, api_key):
    results = []
    used_child_indices = set()
    for cds_sec in cds_sections:
        best_match, score = find_best_match(cds_sec, child_sections)
        if best_match and score > 0.5:
            summary = summarize_section_with_gemini(cds_sec['content'], best_match['content'], api_key)
            results.append({
                'cds_title': cds_sec['title'],
                'child_title': best_match['title'],
                'cds_content': cds_sec['content'],
                'child_content': best_match['content'],
                'summary': summary
            })
            used_child_indices.add(child_sections.index(best_match))
        else:
            summary = summarize_section_with_gemini(cds_sec['content'], '', api_key)
            results.append({
                'cds_title': cds_sec['title'],
                'child_title': None,
                'cds_content': cds_sec['content'],
                'child_content': '',
                'summary': summary
            })
    # Add unmatched child sections
    for idx, child_sec in enumerate(child_sections):
        if idx not in used_child_indices:
            summary = summarize_section_with_gemini('', child_sec['content'], api_key)
            results.append({
                'cds_title': None,
                'child_title': child_sec['title'],
                'cds_content': '',
                'child_content': child_sec['content'],
                'summary': summary
            })
    return results

@app.post("/compare")
async def compare(source: UploadFile = File(...), child: UploadFile = File(...)):
    try:
        # Extract text from uploaded files
        if source.filename.endswith('.pdf'):
            source.file.seek(0)
            cds_text = extract_text_from_pdf(source.file)
        elif source.filename.endswith('.docx'):
            source.file.seek(0)
            cds_text = extract_text_from_docx(source.file)
        else:
            raise ValueError("Unsupported source file type. Only PDF and DOCX are supported.")

        if child.filename.endswith('.pdf'):
            child.file.seek(0)
            child_text = extract_text_from_pdf(child.file)
        elif child.filename.endswith('.docx'):
            child.file.seek(0)
            child_text = extract_text_from_docx(child.file)
        else:
            raise ValueError("Unsupported child file type. Only PDF and DOCX are supported.")

        # Split into sections
        cds_sections = split_into_sections(cds_text)
        child_sections = split_into_sections(child_text)

        # Load Gemini API key
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        print('GEMINI_API_KEY:', os.environ.get('GEMINI_API_KEY'))
        if not gemini_api_key:
            raise ValueError("Gemini API key is not set. Please configure your GEMINI_API_KEY in the .env file.")

        # Match and summarize
        sectionwise_results = match_and_summarize_sections(cds_sections, child_sections, gemini_api_key)

        return {"sections": sectionwise_results}
    except Exception as e:
        print(f"Error in /compare: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload. Please try again later.")
