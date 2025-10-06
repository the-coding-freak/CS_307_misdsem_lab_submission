import re

# ---------- Edit Distance ----------
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


# ---------- Split sentences ----------
def split_sentences(text):
    sentences = re.split(r'[.!?]', text)
    return [s.strip().lower() for s in sentences if s.strip()]


# ---------- Detect Plagiarism ----------
def detect_plagiarism(doc1, doc2, threshold=0.5):
    s1 = split_sentences(doc1)
    s2 = split_sentences(doc2)

    total = len(s1)
    matches = []

    for sentence1 in s1:
        best_match = None
        best_similarity = 0

        for sentence2 in s2:
            distance = edit_distance(sentence1, sentence2)
            max_len = max(len(sentence1), len(sentence2))
            similarity = 1 - distance / max_len if max_len > 0 else 0

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = sentence2

        if best_similarity >= threshold:
            matches.append((sentence1, best_match, best_similarity))

    ppd = (len(matches) / total) * 100 if total > 0 else 0
    return matches, ppd


# ---------- Test Cases ----------
tests = {
    "Test Case 1: Identical Documents": (
        "The sun rises in the east and sets in the west. It provides light and energy for all living beings.",
        "The sun rises in the east and sets in the west. It provides light and energy for all living beings."
    ),

    "Test Case 2: Slightly Modified Document": (
        "The sun rises in the east and sets in the west. It gives warmth and energy to all creatures.",
        "The sun rises in the east and sets in the west. It provides light and energy for all living beings."
    ),

    "Test Case 3: Completely Different Documents": (
        "Artificial intelligence is transforming modern industries with automation and smart analytics.",
        "The rainforest is home to countless species of animals and plants that balance the ecosystem."
    ),

    "Test Case 4: Partial Overlap": (
        "The internet has changed communication across the world. Social media connects people instantly.",
        "The internet has changed communication across the globe. However, excessive use causes isolation."
    )
}


# ---------- Run Tests ----------
for name, (doc1, doc2) in tests.items():
    matches, ppd = detect_plagiarism(doc1, doc2)

    print("=" * 30)
    print(name)
    print("=" * 30)
    print("Potential Plagiarism Detected (Similarity >= 0.5):")

    if matches:
        for s1, s2, sim in matches:
            print(f' - "{s1}" ↔ "{s2}" (Similarity: {sim:.2f})')
    else:
        print(" No matches found")

    print(f"\nPPD Value: {ppd:.2f}%\n")
