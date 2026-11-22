"""All active evaluation inputs are either LegalSeg records or explicitly marked authored examples."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PILOT=ROOT/"gemini-seven-pilot"
LABELS=["none","facts","issue","petitioner_argument","respondent_argument","reasoning","decision"]
FOLDERS={"pilot":"gemini-seven-pilot","confirmation":"gemini-confirmation","paragraphs":"gemini-seven-paragraphs","homogeneous":"gemini-seven-joined"}
def folder_for(name):return ROOT/FOLDERS[name]
