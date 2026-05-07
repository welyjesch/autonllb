#!/usr/bin/env python3
from prepare_sentences import split_into_sentences

def test_initials():
    test_cases = [
        ("John A. Smith is a researcher.", ["John A. Smith is a researcher."]),
        ("Visit Prof. A. Rodriguez for details.", ["Visit Prof. A. Rodriguez for details."]),
        ("The study by J. Doe and M. Smith was great.", ["The study by J. Doe and M. Smith was great."]),
        ("A. Smith and B. Jones are here.", ["A. Smith and B. Jones are here."]),
        ("This is a sentence. A. Smith is here.", ["This is a sentence.", "A. Smith is here."]),
        ("Mr. A. Smith, PhD, is here.", ["Mr. A. Smith, PhD, is here."]),
    ]

    print("Testing Initials Protection...")
    for text, expected in test_cases:
        result = split_into_sentences(text)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] Text: {repr(text)}")
        if status == "FAIL":
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")

if __name__ == "__main__":
    test_initials()
