import re
from typing import List

ABBREVIATIONS = [
    # Version numbers - MORE SPECIFIC patterns first!
    r'\b\d+\.\d+\.\d+\b',  # 3-part version (2.5.1) BEFORE 2-part (2.5)
    r'\b\d+\.\d+\b',       # 2-part version (2.5)
    r'\.\.\.+',
    r'\b\d+\.',
    r'\b[A-Z]\.',
    r'\bDr\.', r'\bMr\.', r'\bMrs\.', r'\bMs\.', r'\bProf\.', 
    r'\bSr\.', r'\bJr\.', r'\bSt\.', r'\bFr\.', r'\bRev\.',
    r'\bGov\.', r'\bSen\.', r'\bRep\.', r'\bPres\.', r'\bGen\.', 
    r'\bCol\.', r'\bLt\.', r'\bSgt\.', r'\bCpl\.', r'\bCapt\.', 
    r'\bCmdr\.', r'\bAdm\.', r'\bAmb\.', r'\bHon\.',
    r'\bPhD\.', r'\bMD\.', r'\bBA\.', r'\bBS\.', r'\bMA\.', r'\bMS\.',
    r'\bInc\.', r'\bLtd\.', r'\bCo\.', r'\bCorp\.', r'\bLLC\.',
    r'\bvs\.', r'\betc\.', r'\bi\.e\.', r'\be\.g\.',
    r'\bFig\.', r'\bfig\.', r'\bVol\.', r'\bvol\.', 
    r'\bNo\.', r'\bno\.', r'\bpp\.', r'\bpg\.', r'\bp\.',
    r'\bed\.', r'\beds\.', r'\best\.', r'\bapprox\.', 
    r'\bca\.', r'\bcf\.', r'\bviz\.', r'\bal\.', r'\bet al\.',
    # NOTE: Don't add comma-prefixed patterns like r',\s*etc\.' - they break sentence splitting!
    # The \betc\. pattern already matches "etc." after commas via word boundary
]

ABBREVIATION_PATTERN = '(' + '|'.join(ABBREVIATIONS) + r')'

def split_into_sentences_fixed(text: str) -> List[str]:
    """Split text into sentences using regex pattern with fixed protect_abbreviation."""
    PLACEHOLDER = '<<ABBR_PERIOD>>'
    
    def protect_abbreviation(match):
        abbrev = match.group(0)
        
        # NEVER replace "etc." - it should never be a sentence boundary
        if abbrev == 'etc.':
            return abbrev
        
        # PROTECT ELLIPSIS - do not replace dots in "..."
        if '...' in abbrev:
            return abbrev
        
        end_pos = match.end()
        
        # Don't replace when followed by comma
        if end_pos < len(text) and text[end_pos] == ',':
            return abbrev
        
        # Don't replace when at end of string
        if end_pos >= len(text):
            return abbrev
        
        # Don't replace if followed by space and then an uppercase letter (i.e., sentence boundary)
        if end_pos < len(text) and text[end_pos] == ' ':
            # Look ahead for the next non-space character
            next_non_space = end_pos + 1
            while next_non_space < len(text) and text[next_non_space] == ' ':
                next_non_space += 1
            if next_non_space < len(text) and text[next_non_space].isupper():
                return abbrev
        
        return abbrev.replace('.', PLACEHOLDER)
    
    protected_text = re.sub(ABBREVIATION_PATTERN, protect_abbreviation, text)
    
    # Split on period followed by space + capital letter
    # The negative lookbehind ensures we don't split after a letter (like in abbreviations)
    sentences = re.split(r'(?<=[.!?])\s+', protected_text)
    sentences = [s.replace(PLACEHOLDER, '.') for s in sentences]
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if len(s) >= 10]
    
    return sentences

text = (
    "Dr. James Mitchell, PhD. in Computer Science, led the research team at Stanford Inc. "
    "The initial findings showed improvement of 3.5% on the benchmark. "
    "Prof. Elena Rodriguez, working with Mr. Chen and Ms. Taylor, published results in Nature Vol. 42 (pp. 100-110). "
    "See Fig. 4 et al. for detailed comparisons vs. previous work i.e. from 2023."
)

sentences = split_into_sentences_fixed(text)
print(f"Number of sentences: {len(sentences)}")
for i, s in enumerate(sentences):
    print(f"Sentence {i}: {repr(s)}")
    print(f"  Length: {len(s)}")