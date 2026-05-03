import sys
sys.path.insert(0, '.')

# Import the split_into_sentences from test_parser
from test_parser import split_into_sentences

text = (
    "Dr. James Mitchell, PhD. in Computer Science, led the research team at Stanford Inc. "
    "The initial findings showed improvement of 3.5% on the benchmark. "
    "Prof. Elena Rodriguez, working with Mr. Chen and Ms. Taylor, published results in Nature Vol. 42 (pp. 100-110). "
    "See Fig. 4 et al. for detailed comparisons vs. previous work i.e. from 2023."
)

sentences = split_into_sentences(text)
print(f"Number of sentences: {len(sentences)}")
for i, s in enumerate(sentences):
    print(f"Sentence {i}: {repr(s)}")
    print(f"  Length: {len(s)}")