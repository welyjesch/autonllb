import re

ABBREVIATIONS = [
    r'\b\d+\.\d+\.\d+\b',
    r'\b\d+\.\d+\b',
    r'\.\.\.+',
    r'\b\d+\.',
    r'\b[A-Z]\.',
]

pattern = '(' + '|'.join(ABBREVIATIONS) + r')'
print("Pattern:", pattern)

text = "3.5%"
matches = list(re.finditer(pattern, text))
print("Matches in", repr(text))
for m in matches:
    print(f"  {m.group()} at {m.start()}-{m.end()}")

# Also test with the full context
text2 = "improvement of 3.5% on the benchmark"
matches2 = list(re.finditer(pattern, text2))
print("\nMatches in", repr(text2))
for m in matches2:
    print(f"  {m.group()} at {m.start()}-{m.end()}")