"""category classifier — assigns documents and queries to one of 5 categories.

uses keyword matching with fallback heuristics. no LLM needed.
"""

import re
from typing import Optional

CATEGORIES = ["personal", "work", "documents", "coding", "chat_logs"]

# keyword patterns for each category (checked in order, first match wins)
CATEGORY_KEYWORDS = {
    "coding": {
        "strong": [
            "function", "class", "def ", "import ", "const ", "let ", "var ",
            "return", "async", "await", "lambda", "struct", "enum",
            "git", "commit", "branch", "merge", "pull request",
            "python", "javascript", "typescript", "rust", "golang", "c++",
            "api", "endpoint", "http", "json", "yaml",
            "docker", "kubernetes", "nginx", "systemd",
            "bug", "debug", "error", "exception", "stack trace",
            "test", "unittest", "pytest", "jest",
            "database", "sql", "query", "index",
            "compile", "build", "deploy", "ci/cd",
        ],
        "patterns": [
            r"\b(def|class|function|const|let|var)\s+\w+",
            r"\b(import|from|require|include)\s+",
            r"\.(py|js|ts|cpp|rs|go|java|rb|sql)\b",
            r"```\w*\n",  # code blocks
        ],
    },
    "chat_logs": {
        "strong": [
            "conversation", "chat", "session", "transcript",
            "claude", "openai", "llm", "chatbot",
            "user:", "assistant:", "[user]", "[assistant]",
            "tokens", "context window", "prompt",
        ],
        "patterns": [
            r"\[(user|assistant|system)\]:",
            r"^(user|assistant):\s",
            r"session[_-]?id",
            r"conversation[_-]?id",
        ],
    },
    "work": {
        "strong": [
            "meeting", "project", "deadline", "milestone", "sprint",
            "client", "customer", "stakeholder", "deliverable",
            "revenue", "profit", "budget", "invoice", "contract",
            "team", "manager", "lead", "hire", "onboard",
            "roadmap", "strategy", "objective", "kpi", "metric",
            "chainlinks", "chain", "consulting", "business",
            "proposal", "sow", "nda", "agreement",
        ],
        "patterns": [
            r"\b(Q[1-4]|FY\d{2,4})\b",
            r"\b(ROI|KPI|OKR|P&L)\b",
        ],
    },
    "personal": {
        "strong": [
            "journal", "diary", "personal", "thought", "feeling",
            "health", "workout", "exercise", "diet", "sleep",
            "family", "friend", "birthday", "vacation", "hobby",
            "goal", "habit", "gratitude", "reflection",
            "recipe", "shopping", "todo", "reminder",
        ],
        "patterns": [
            r"\b(I feel|I think|I want|I need|my goal)\b",
        ],
    },
    "documents": {
        "strong": [
            "report", "paper", "article", "research", "study",
            "pdf", "document", "chapter", "section", "appendix",
            "abstract", "conclusion", "methodology", "findings",
            "reference", "citation", "bibliography",
            "manual", "guide", "handbook", "specification",
            "license", "terms", "policy", "regulation",
        ],
        "patterns": [
            r"\b(Figure|Table|Appendix)\s+\d+",
            r"\b(Section|Chapter)\s+\d+",
        ],
    },
}

# weights for scoring
STRONG_KEYWORD_WEIGHT = 2
PATTERN_MATCH_WEIGHT = 3


def classify(text: str, filename: Optional[str] = None) -> str:
    """classify text into one of the 5 categories.

    returns the category name. falls back to 'documents' if no clear match.
    """
    text_lower = text.lower()
    scores = {cat: 0 for cat in CATEGORIES}

    # filename hints
    if filename:
        fn_lower = filename.lower()
        if any(ext in fn_lower for ext in [".py", ".js", ".ts", ".cpp", ".rs", ".go", ".java", ".rb", ".sql", ".sh"]):
            scores["coding"] += 10
        elif "chat" in fn_lower or "conversation" in fn_lower or "session" in fn_lower:
            scores["chat_logs"] += 10
        elif any(ext in fn_lower for ext in [".pdf", ".docx", ".doc", ".xlsx"]):
            scores["documents"] += 10
        elif "journal" in fn_lower or "diary" in fn_lower or "personal" in fn_lower:
            scores["personal"] += 10

    # keyword matching
    for cat, rules in CATEGORY_KEYWORDS.items():
        for kw in rules["strong"]:
            count = text_lower.count(kw.lower())
            if count > 0:
                scores[cat] += count * STRONG_KEYWORD_WEIGHT

        for pattern in rules["patterns"]:
            matches = re.findall(pattern, text_lower if "^" not in pattern else text, re.MULTILINE | re.IGNORECASE)
            scores[cat] += len(matches) * PATTERN_MATCH_WEIGHT

    # pick highest score
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "documents"  # fallback
    return best


def classify_query(query: str) -> list[str]:
    """classify a search query, returning categories in priority order.

    queries might span multiple categories, so return ranked list.
    """
    query_lower = query.lower()
    scores = {cat: 0 for cat in CATEGORIES}

    for cat, rules in CATEGORY_KEYWORDS.items():
        for kw in rules["strong"]:
            if kw.lower() in query_lower:
                scores[cat] += STRONG_KEYWORD_WEIGHT

        for pattern in rules["patterns"]:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            scores[cat] += len(matches) * PATTERN_MATCH_WEIGHT

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # return categories with score > 0, or just documents as fallback
    result = [cat for cat, score in ranked if score > 0]
    if not result:
        result = ["documents"]
    return result
