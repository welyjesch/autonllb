import re

# Recreate the protected text exactly as in debug_detailed.py
ABBREVIATIONS = [
    r'\b\d+\.\d+\.\d+\b',
    r'\b\d+\.\d+\b',
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
]
ABBREVIATION_PATTERN = '(' + '|'.join(ABBREVIATIONS) + r')'
PLACEHOLDER = '<<ABBR_PERIOD>>'

def protect_abbreviation(match):
    abbrev = match.group(0)
    if abbrev == 'etc.':
        return abbrev
    if '...' in abbrev:
        return abbrev
    # We don't have the original text here, so we'll skip the comma and end checks for simplicity
    # In the real function, these checks depend on the original text
    return abbrev.replace('.', PLACEHOLDER)

text = (
    "Dr. James Mitchell, PhD. in Computer Science, led the research team at Stanford Inc. "
    "The initial findings showed improvement of 3.5% on the benchmark. "
    "Prof. Elena Rodriguez, working with Mr. Chen and Ms. Taylor, published results in Nature Vol. 42 (pp. 100-110). "
    "See Fig. 4 et al. for detailed comparisons vs. previous work i.e. from 2023."
)

protected_text = re.sub(ABBREVIATION_PATTERN, protect_abbreviation, text)
print("Protected text:")
print(repr(protected_text))
print()

# Find the position of "benchmark. Prof"
import re
pattern = r'benchmark\. Prof'
match = re.search(pattern, protected_text)
if match:
    print(f"Found 'benchmark. Prof' at index {match.start()}")
    # Show context
    start = max(0, match.start() - 10)
    end = min(len(protected_text), match.end() + 10)
    print(f"Context: {repr(protected_text[start:end])}")
    # Check the character before the space and the space
    # In the substring "benchmark. Prof", the space is at index 9 of the substring (0-based)
    # In the full string, the space is at match.start() + 9
    space_index = match.start() + 9  # "benchmark." is 10 chars, then space
    print(f"Space index: {space_index}")
    print(f"Character before space: {repr(protected_text[space_index-1])}")
    print(f"Character at space: {repr(protected_text[space_index])}")
    print(f"Character after space: {repr(protected_text[space_index+1])}")
else:
    print("'benchmark. Prof' not found")

# Now try to split
print("\nTrying to split the protected text with r'(?<=[.!?])\\s+'")
split_result = re.split(r'(?<=[.!?])\s+', protected_text)
print(f"Number of splits: {len(split_result)}")
for i, s in enumerate(split_result):
    print(f"  {i}: {repr(s[:100])}{'...' if len(s) > 100 else ''}")