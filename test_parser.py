#!/usr/bin/env python3
"""
Unit Tests for Sentence Splitting with Abbreviation Handling
Tests the split_into_sentences function - standalone version
"""

import re
from typing import List


# ==============================================================================
# Sentence Splitting Logic (copied from prepare_data.py to avoid dependencies)
# ==============================================================================

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


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex pattern."""
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
        
        return abbrev.replace('.', PLACEHOLDER)
    
    protected_text = re.sub(ABBREVIATION_PATTERN, protect_abbreviation, text)
    
    # Split on period followed by space + capital letter
    # The negative lookbehind ensures we don't split after a letter (like in abbreviations)
    sentences = re.split(r'(?<=[.!?])\s+', protected_text)
    sentences = [s.replace(PLACEHOLDER, '.') for s in sentences]
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if len(s) >= 10]
    
    return sentences


def assert_equal(actual, expected, test_name):
    """Simple assertion helper."""
    if actual == expected:
        print(f"  ✓ {test_name}")
        return True
    else:
        print(f"  ✗ {test_name}")
        print(f"    Expected: {expected}")
        print(f"    Got:      {actual}")
        return False


def assert_true(condition, test_name):
    """Simple boolean assertion."""
    if condition:
        print(f"  ✓ {test_name}")
        return True
    else:
        print(f"  ✗ {test_name}")
        return False


def test_title_abbreviations():
    """Test common title abbreviations."""
    print("\n[TITLE ABBREVIATIONS]")
    text = "Dr. Smith went to the hospital. Mr. Johnson called him."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 2, "Split into 2 sentences")
    assert_true("Dr. Smith" in sentences[0], "First sentence contains 'Dr. Smith'")
    assert_true("Mr. Johnson" in sentences[1], "Second sentence contains 'Mr. Johnson'")


def test_multiple_title_abbreviations():
    """Test multiple abbreviations in one sentence."""
    print("\n[MULTIPLE TITLE ABBREVIATIONS]")
    text = "Ms. Garcia and Mrs. Chen met with Prof. Anderson."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 1, "Single sentence")
    assert_true("Ms. Garcia" in sentences[0], "Contains 'Ms. Garcia'")
    assert_true("Mrs. Chen" in sentences[0], "Contains 'Mrs. Chen'")
    assert_true("Prof. Anderson" in sentences[0], "Contains 'Prof. Anderson'")


def test_military_ranks():
    """Test military rank abbreviations."""
    print("\n[MILITARY RANKS]")
    text = "Gen. Lee commanded the troops. Col. Martinez was his assistant. Lt. Johnson followed orders."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 3, "Split into 3 sentences")
    assert_true("Gen. Lee" in sentences[0], "First sentence contains 'Gen. Lee'")
    assert_true("Col. Martinez" in sentences[1], "Second sentence contains 'Col. Martinez'")
    assert_true("Lt. Johnson" in sentences[2], "Third sentence contains 'Lt. Johnson'")


def test_professional_degrees():
    """Test professional degree abbreviations."""
    print("\n[PROFESSIONAL DEGREES]")
    text = "Dr. Sarah Smith, PhD. in Biology, received her MD. from the university."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 1, "Single sentence")
    assert_true("PhD." in sentences[0], "Contains 'PhD.'")
    assert_true("MD." in sentences[0], "Contains 'MD.'")


def test_business_abbreviations():
    """Test business entity abbreviations."""
    print("\n[BUSINESS ABBREVIATIONS]")
    text = "Apple Inc. is headquartered in California. Google LLC. operates worldwide. Microsoft Corp. has offices globally."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 3, "Split into 3 sentences")
    assert_true("Inc." in sentences[0], "First sentence contains 'Inc.'")
    assert_true("LLC." in sentences[1], "Second sentence contains 'LLC.'")
    assert_true("Corp." in sentences[2], "Third sentence contains 'Corp.'")


def test_decimal_numbers():
    """Test decimal point preservation in numbers."""
    print("\n[DECIMAL NUMBERS]")
    text = "The temperature was 37.5 degrees. It rose to 38.2 by afternoon."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 2, "Split into 2 sentences")
    assert_true("37.5" in sentences[0], "First sentence contains '37.5'")
    assert_true("38.2" in sentences[1], "Second sentence contains '38.2'")


def test_numbered_lists():
    """Test numbered lists with periods."""
    print("\n[NUMBERED LISTS]")
    text = "First item. Second item. Third item."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 3, "Split into 3 sentences")


def test_version_numbers():
    """Test version number format (e.g., 1.2.3)."""
    print("\n[VERSION NUMBERS]")
    text = "The software version 2.5.1 was released. Version 2.6.0 came next."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 2, "Split into 2 sentences")
    assert_true("2.5.1" in sentences[0], "First sentence contains '2.5.1'")
    assert_true("2.6.0" in sentences[1], "Second sentence contains '2.6.0'")


def test_money_amounts():
    """Test monetary values with decimals."""
    print("\n[MONEY AMOUNTS]")
    text = "The cost was $19.99. The tax was $2.50."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 2, "Split into 2 sentences")
    assert_true("19.99" in sentences[0], "First sentence contains '19.99'")
    assert_true("2.50" in sentences[1], "Second sentence contains '2.50'")


def test_latin_abbreviations():
    """Test Latin abbreviations (e.g., et al., i.e., e.g.)."""
    print("\n[LATIN ABBREVIATIONS]")
    text = "Multiple authors et al. conducted research. For example e.g. they studied behavior. That is i.e. they examined patterns."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 3, "Split into 3 sentences")
    assert_true("et al." in sentences[0], "First sentence contains 'et al.'")
    assert_true("e.g." in sentences[1], "Second sentence contains 'e.g.'")
    assert_true("i.e." in sentences[2], "Third sentence contains 'i.e.'")


def test_vs_abbreviation():
    """Test 'vs.' abbreviation."""
    print("\n[VS ABBREVIATION]")
    text = "Team A vs. Team B competed. Player X vs. Player Y faced off."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 2, "Split into 2 sentences")
    assert_true("vs." in sentences[0], "First sentence contains 'vs.'")
    assert_true("vs." in sentences[1], "Second sentence contains 'vs.'")


def test_etc_abbreviation():
    """Test 'etc.' abbreviation."""
    print("\n[ETC ABBREVIATION]")
    text = "He liked apples, oranges, etc. She enjoyed reading books, etc."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 2, "Split into 2 sentences")
    assert_true("etc." in sentences[0], "First sentence contains 'etc.'")
    assert_true("etc." in sentences[1], "Second sentence contains 'etc.'")


def test_figure_and_volume():
    """Test figure and volume abbreviations."""
    print("\n[FIGURE AND VOLUME]")
    text = "See Fig. 3 for details. Check Vol. 2 of the manual."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 2, "Split into 2 sentences")
    assert_true("Fig." in sentences[0], "First sentence contains 'Fig.'")
    assert_true("Vol." in sentences[1], "Second sentence contains 'Vol.'")


def test_multiple_periods():
    """Test ellipsis (...) handling."""
    print("\n[ELLIPSIS]")
    text = "He said goodbye... Then he left."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 2, "Split into 2 sentences")
    assert_true("goodbye" in sentences[0], "First sentence contains 'goodbye'")


def test_exclamation_and_question():
    """Test mixed punctuation marks."""
    print("\n[MIXED PUNCTUATION]")
    text = "Watch out! Is he coming? Yes, he is!"
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 3, "Split into 3 sentences")


def test_empty_string():
    """Test empty input."""
    print("\n[EMPTY STRING]")
    text = ""
    sentences = split_into_sentences(text)
    assert_equal(sentences, [], "Empty input returns empty list")


def test_single_sentence():
    """Test single sentence."""
    print("\n[SINGLE SENTENCE]")
    text = "This is a single sentence."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 1, "Single sentence")
    assert_equal(sentences[0], "This is a single sentence.", "Sentence content matches")


def test_sentence_too_short():
    """Test that very short sentences are filtered out."""
    print("\n[SHORT SENTENCE FILTERING]")
    text = "Short. This is a longer sentence that should be kept."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 1, "Short sentence filtered out")
    assert_true("longer sentence" in sentences[0], "Longer sentence preserved")


def test_whitespace_handling():
    """Test multiple whitespaces between sentences."""
    print("\n[WHITESPACE HANDLING]")
    text = "First sentence.    Second sentence.  Third sentence."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 3, "Split into 3 sentences")
    for i, sent in enumerate(sentences):
        is_stripped = sent == sent.strip()
        assert_true(is_stripped, f"Sentence {i+1} is stripped")


def test_newlines_and_tabs():
    """Test handling of newlines and tabs."""
    print("\n[NEWLINES AND TABS]")
    text = "First sentence.\nSecond sentence.\tThird sentence."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 3, "Split into 3 sentences")


def test_academic_text():
    """Test academic text with abbreviations and citations."""
    print("\n[ACADEMIC TEXT]")
    text = "Dr. Smith published in Nature Vol. 5 (p. 123-125). Fig. 2 shows the results. See et al. for more details."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 3, "Split into 3 sentences")
    has_dr = any("Dr." in s for s in sentences)
    has_vol = any("Vol." in s for s in sentences)
    assert_true(has_dr, "Contains 'Dr.'")
    assert_true(has_vol, "Contains 'Vol.'")


def test_news_article():
    """Test news article format."""
    print("\n[NEWS ARTICLE]")
    text = "Mr. Johnson, CEO of Tech Inc., announced layoffs. The company saw revenue of $2.5B last year. Ms. Garcia responded saying this would affect 1,000+ workers."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 3, "Split into 3 sentences")
    has_inc = any("Inc." in s for s in sentences)
    has_number = any("2.5" in s for s in sentences)
    assert_true(has_inc, "Contains 'Inc.'")
    assert_true(has_number, "Contains decimal numbers")


def test_dialogue_with_titles():
    """Test dialogue with speaker titles."""
    print("\n[DIALOGUE WITH TITLES]")
    text = "Prof. Anderson asked the class. Dr. Wilson provided an answer. Mr. Chen added his thoughts."
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 3, "Split into 3 sentences")
    has_prof = any("Prof." in s for s in sentences)
    assert_true(has_prof, "Contains 'Prof.'")


def test_mixed_punctuation_with_abbreviations():
    """Test mixed punctuation with multiple abbreviations."""
    print("\n[MIXED PUNCTUATION WITH ABBREVIATIONS]")
    text = "Did Dr. Smith say he works for Apple Inc.? Mr. Johnson thought so. Ms. Chen wasn't sure!"
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 3, "Split into 3 sentences")
    has_dr = any("Dr." in s for s in sentences)
    has_inc = any("Inc." in s for s in sentences)
    assert_true(has_dr, "Contains 'Dr.'")
    assert_true(has_inc, "Contains 'Inc.'")


def test_long_complex_text():
    """Test longer, more realistic text."""
    print("\n[LONG COMPLEX TEXT]")
    text = (
        "Dr. James Mitchell, PhD. in Computer Science, led the research team at Stanford Inc. "
        "The initial findings showed improvement of 3.5% on the benchmark. "
        "Prof. Elena Rodriguez, working with Mr. Chen and Ms. Taylor, published results in Nature Vol. 42 (pp. 100-110). "
        "See Fig. 4 et al. for detailed comparisons vs. previous work i.e. from 2023."
    )
    sentences = split_into_sentences(text)
    assert_equal(len(sentences), 4, "Split into 4 sentences")
    all_long = all(len(s) >= 10 for s in sentences)
    assert_true(all_long, "All sentences are at least 10 chars")
    has_phd = any("PhD." in s for s in sentences)
    has_inc = any("Inc." in s for s in sentences)
    has_decimal = any("3.5" in s for s in sentences)
    assert_true(has_phd, "Contains 'PhD.'")
    assert_true(has_inc, "Contains 'Inc.'")
    assert_true(has_decimal, "Contains '3.5'")


def test_abbreviation_preserved_in_output():
    """Test that abbreviations are fully preserved."""
    print("\n[ABBREVIATION PRESERVATION]")
    text = "Contact Dr. Anderson at the hospital."
    sentences = split_into_sentences(text)
    assert_true("Dr." in sentences[0], "Contains 'Dr.'")
    assert_true("Anderson" in sentences[0], "Contains 'Anderson'")


def test_numbers_preserved():
    """Test that decimal numbers are fully preserved."""
    print("\n[NUMBER PRESERVATION]")
    text = "The ratio was 2.71828 according to the formula."
    sentences = split_into_sentences(text)
    assert_true("2.71828" in sentences[0], "Contains '2.71828'")


def test_all_content_preserved():
    """Test that no content is lost in splitting."""
    print("\n[CONTENT PRESERVATION]")
    text = "Dr. Smith, MD., calculated 3.14159. He published in Nature Vol. 5 et al."
    sentences = split_into_sentences(text)
    full_text = " ".join(sentences)
    
    assert_true("Dr." in full_text, "Contains 'Dr.'")
    assert_true("Smith" in full_text, "Contains 'Smith'")
    assert_true("MD." in full_text, "Contains 'MD.'")
    assert_true("3.14159" in full_text, "Contains '3.14159'")
    assert_true("Vol." in full_text, "Contains 'Vol.'")


# ==============================================================================
# Main Test Runner
# ==============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("SENTENCE SPLITTING TEST SUITE")
    print("=" * 70)
    
    test_title_abbreviations()
    test_multiple_title_abbreviations()
    test_military_ranks()
    test_professional_degrees()
    test_business_abbreviations()
    test_decimal_numbers()
    test_numbered_lists()
    test_version_numbers()
    test_money_amounts()
    test_latin_abbreviations()
    test_vs_abbreviation()
    test_etc_abbreviation()
    test_figure_and_volume()
    test_multiple_periods()
    test_exclamation_and_question()
    test_empty_string()
    test_single_sentence()
    test_sentence_too_short()
    test_whitespace_handling()
    test_newlines_and_tabs()
    test_academic_text()
    test_news_article()
    test_dialogue_with_titles()
    test_mixed_punctuation_with_abbreviations()
    test_long_complex_text()
    test_abbreviation_preserved_in_output()
    test_numbers_preserved()
    test_all_content_preserved()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
