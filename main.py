import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import re
import json
import difflib

def fuzzy_match(a, b):
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    return a == b or a in b or b in a

def normalize_title(title):
    # Remove numbers, punctuation, and convert to uppercase
    return re.sub(r'[^A-Z ]', '', title.upper()) if title else ''

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://delightful-sky-044169710.2.azurestaticapps.net",  # Azure frontend
        "http://localhost:5173",  # Local development (optional)
        "http://127.0.0.1:5173"   # Local development (optional)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/compare")
async def compare(source: UploadFile = File(...), child: UploadFile = File(...)):
    try:
        # Read file contents
        source.file.seek(0)
        cds_text = source.file.read().decode(errors='ignore') if source.filename.endswith('.txt') else source.file.read().decode(errors='ignore')
        child.file.seek(0)
        child_text = child.file.read().decode(errors='ignore') if child.filename.endswith('.txt') else child.file.read().decode(errors='ignore')

        # If PDF or DOCX, use PyPDF2 or python-docx to extract text
        if source.filename.endswith('.pdf'):
            import PyPDF2
            source.file.seek(0)
            reader = PyPDF2.PdfReader(source.file)
            cds_text = "".join([page.extract_text() or "" for page in reader.pages])
        elif source.filename.endswith('.docx'):
            from docx import Document
            source.file.seek(0)
            doc = Document(source.file)
            cds_text = "\n".join([para.text for para in doc.paragraphs])

        if child.filename.endswith('.pdf'):
            import PyPDF2
            child.file.seek(0)
            reader = PyPDF2.PdfReader(child.file)
            child_text = "".join([page.extract_text() or "" for page in reader.pages])
        elif child.filename.endswith('.docx'):
            from docx import Document
            child.file.seek(0)
            doc = Document(child.file)
            child_text = "\n".join([para.text for para in doc.paragraphs])

        # Gemini API key
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise HTTPException(status_code=500, detail="Gemini API key is not set.")
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        # 1. Gemini call: extract sections from both files
        extraction_prompt = f"""
You are an expert in document analysis. Here are two documents: a Core Data Sheet (CDS) and a Child (USPI) document.
For each document:
1. Split the text into logical sections based on headings or numbering.
2. For each section, return the section title and the section content.
3. Output the result as a JSON object with two keys: \"cds_sections\" and \"child_sections\".
   - \"cds_sections\" should be a list of objects with \"title\" and \"content\" for each CDS section.
   - \"child_sections\" should be a list of objects with \"title\" and \"content\" for each Child section.
Respond ONLY with a valid JSON object, no explanation, no markdown, no extra text, no code block markers.

CDS Document:
{cds_text}

Child Document:
{child_text}
"""
        extraction_response = model.generate_content(extraction_prompt)
        print("Gemini extraction raw response:", repr(extraction_response.text))
        match = re.search(r'\{[\s\S]+\}', extraction_response.text)
        extraction_json = match.group(0) if match else extraction_response.text
        try:
            data = json.loads(extraction_json)
            cds_sections = data.get("cds_sections", [])
            child_sections = data.get("child_sections", [])
        except Exception as e:
            print("Failed to parse Gemini extraction response as JSON:", e)
            cds_sections = []
            child_sections = []

        # Use Gemini for content similarity matching in one call
        matching_prompt = f"""
You are an expert in regulatory document analysis.
Given these two lists of sections, match each section from the USPI list to the most similar section from the CDS list, based on meaning and context (not just wording).
Return a JSON array of objects: {{cds_title, cds_content, child_title, child_content, similarity_score (0-1)}}.
If no good match exists (similarity_score < 0.3), set cds_title and cds_content to empty strings.
CDS Sections:
{json.dumps(cds_sections, ensure_ascii=False)}
USPI Sections:
{json.dumps(child_sections, ensure_ascii=False)}
"""
        matching_response = model.generate_content(matching_prompt)
        print("Gemini matching raw response:", repr(matching_response.text))
        match = re.search(r'\[.*\]', matching_response.text, re.DOTALL)
        matching_json = match.group(0) if match else matching_response.text
        try:
            matched_pairs = json.loads(matching_json)
        except Exception as e:
            print("Failed to parse Gemini matching response as JSON:", e)
            matched_pairs = []

        # Filter matched_pairs to only include real matches (both titles non-empty)
        filtered_matched_pairs = [pair for pair in matched_pairs if pair.get('cds_title') and pair.get('child_title')]

        # Build unified list for all sections (matched and unmatched)
        matched_cds_titles = set(pair['cds_title'] for pair in filtered_matched_pairs)
        matched_child_titles = set(pair['child_title'] for pair in filtered_matched_pairs)
        matched_by_cds_title = {pair['cds_title']: pair for pair in filtered_matched_pairs}
        matched_by_child_title = {pair['child_title']: pair for pair in filtered_matched_pairs}
        unified_list = []
        # Add all CDS sections (matched or unmatched)
        for cds_sec in cds_sections:
            if cds_sec.get('title', '') in matched_by_cds_title:
                pair = matched_by_cds_title[cds_sec.get('title', '')]
                unified_list.append({
                    'cds_title': pair['cds_title'],
                    'cds_content': pair['cds_content'],
                    'child_title': pair['child_title'],
                    'child_content': pair['child_content'],
                    'similarity': pair.get('similarity_score'),
                    'status': 'Matched'
                })
            else:
                unified_list.append({
                    'cds_title': cds_sec.get('title', ''),
                    'cds_content': cds_sec.get('content', ''),
                    'child_title': '',
                    'child_content': '',
                    'similarity': None,
                    'status': 'Unmatched'
                })
        # Add unmatched USPI sections
        for child_sec in child_sections:
            if child_sec.get('title', '') not in matched_by_child_title:
                unified_list.append({
                    'cds_title': '',
                    'cds_content': '',
                    'child_title': child_sec.get('title', ''),
                    'child_content': child_sec.get('content', ''),
                    'similarity': None,
                    'status': 'Unmatched'
                })

        # Build section_comparisons for overall/section-wise summary
        section_comparisons = []
        for pair in filtered_matched_pairs:
            cds_content = pair.get('cds_content', '')
            child_content = pair.get('child_content', '')
            # Use difflib to count changes (number of differing lines)
            cds_lines = cds_content.splitlines()
            child_lines = child_content.splitlines()
            diff = list(difflib.unified_diff(cds_lines, child_lines))
            change_count = sum(1 for line in diff if line.startswith('+ ') or line.startswith('- '))
            # Use Gemini to generate a summary of the difference
            summary_prompt = f"""
You are an expert in regulatory document analysis. Compare the following two sections and summarize the key differences in 1-2 sentences for a regulatory professional. Be concise and specific.
CDS Section:
{cds_content}
USPI Section:
{child_content}
"""
            summary = ""
            try:
                summary_response = model.generate_content(summary_prompt)
                summary = summary_response.text.strip()
            except Exception as e:
                print("Failed to get Gemini summary for section:", e)
                summary = "Summary not available."
            section_comparisons.append({
                'title': pair.get('cds_title') or pair.get('child_title'),
                'change_count': change_count,
                'summary': summary
            })

        return {
            "matched_pairs": filtered_matched_pairs,
            "cds_sections": cds_sections,
            "child_sections": child_sections,
            "unified_list": unified_list,
            "section_comparisons": section_comparisons
        }
    except Exception as e:
        print(f"Error in /compare: {e}")
        raise HTTPException(status_code=500, detail= "Failed to upload. Please try again later")
