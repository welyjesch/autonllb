#!/usr/bin/env python3
import re

# Test the split pattern directly
text = "benchmark. Prof"
print("Testing split on:", repr(text))
result = re.split(r'(?<=[.!?])\s+', text)
print("Split result:", result)

# Test with the actual protected text snippet
protected_snippet = "benchmark. Prof<<ABBR_PERIOD>> Elena Rodriguez"
print("\nTesting split on protected snippet:", repr(protected_snippet))
result2 = re.split(r'(?<=[.!?])\s+', protected_snippet)
print("Split result:", result2)

# Test with what we actually have in the protected text
# From debug: "...on the benchmark. Prof<<ABBR_PERIOD>> Elena..."
# Let's extract the exact part
full_protected = (
    "Dr<<ABBR_PERIOD>> James Mitchell, PhD<<ABBR_PERIOD>> in Computer Science, led the research team at Stanford Inc<<ABBR_PERIOD>> "
    "The initial findings showed improvement of 3<<ABBR_PERIOD>>5% on the benchmark. "
    "Prof<<ABBR_PERIOD>> Elena Rodriguez, working with Mr<<ABBR_PERIOD>> Chen and Ms<<ABBR_PERIOD>> Taylor, published results in Nature Vol<<ABBR_PERIOD>> 42 (pp<<ABBR_PERIOD>> 100-110). "
    "See Fig<<ABBR_PERIOD>> 4 et al<<ABBR_PERIOD>> for detailed comparisons vs<<ABBR_PERIOD>> previous work i<<ABBR_PERIOD>>e<<ABBR_PERIOD>> from 2023<<ABBR_PERIOD>>"
)

print("\nFull protected text split:")
split_full = re.split(r'(?<=[.!?])\s+', full_protected)
for i, s in enumerate(split_full):
    print(f"  {i}: {repr(s[:80])}{'...' if len(s) > 80 else ''}")