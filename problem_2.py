import heapq
import re
import itertools

# ---------- Preprocessing ----------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    sentences = text.split(".")
    return [s.strip() for s in sentences if s.strip()]

# ---------- Edit Distance ----------
def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,     # deletion
                           dp[i][j - 1] + 1,     # insertion
                           dp[i - 1][j - 1] + cost)  # substitution
    return dp[m][n]

# ---------- Heuristic ----------
def heuristic(i, j, D1, D2):
    remaining = min(len(D1) - i, len(D2) - j)
    return remaining  # assume each remaining alignment costs at least 1

# ---------- A* Algorithm ----------
def a_star_alignment(D1, D2):
    m, n = len(D1), len(D2)
    start = (0, 0, 0, [])  # (i, j, g, path)
    open_list = []
    visited = {}
    SKIP_COST = 5
    counter = itertools.count()  # unique tie-breaker

    # push the start node
    heapq.heappush(open_list, (0, next(counter), start))

    while open_list:
        f, _, (i, j, g, path) = heapq.heappop(open_list)

        if (i, j) in visited and visited[(i, j)] <= g:
            continue
        visited[(i, j)] = g

        if i == m and j == n:
            return path

        # Align sentences
        if i < m and j < n:
            cost = levenshtein(D1[i], D2[j])
            new_state = (i + 1, j + 1, g + cost, path + [(D1[i], D2[j], cost)])
            heapq.heappush(open_list, (new_state[2] + heuristic(i + 1, j + 1, D1, D2), next(counter), new_state))

        # Skip sentence in D1
        if i < m:
            new_state = (i + 1, j, g + SKIP_COST, path + [(D1[i], None, SKIP_COST)])
            heapq.heappush(open_list, (new_state[2] + heuristic(i + 1, j, D1, D2), next(counter), new_state))

        # Skip sentence in D2
        if j < n:
            new_state = (i, j + 1, g + SKIP_COST, path + [(None, D2[j], SKIP_COST)])
            heapq.heappush(open_list, (new_state[2] + heuristic(i, j + 1, D1, D2), next(counter), new_state))

    return None


# ---------- Run Example ----------
if _name_ == "_main_":
    # Input file names
    file1 = input("Enter first file name: ")
    file2 = input("Enter second file name: ")

    # Read file contents
    with open(file1, 'r', encoding='utf-8') as f1:
        doc1 = f1.read()

    with open(file2, 'r', encoding='utf-8') as f2:
        doc2 = f2.read()

    # Preprocess documents
    D1 = preprocess(doc1)
    D2 = preprocess(doc2)

    # Run A* alignment
    result = a_star_alignment(D1, D2)

    # Display results
    print("\nAlignment Results:")
    for pair in result:
        print(pair)

    print("\nPotential Plagiarism Detected (Edit Distance <= 3):")
    for s1, s2, cost in result:
        if s1 and s2 and cost <= 3:
            print(f' - "{s1}" ↔ "{s2}" (Edit Distance: {cost})')
