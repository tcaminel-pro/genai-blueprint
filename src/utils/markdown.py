import re
from typing import List, Tuple

"""
The main logic was inspired by the discussions and patterns listed in
stackoverflow.com
and the Markdown-regex gist by gist.github.com/elfefe.
"""

# ------------------------------------------------------------
# Helper regexes for the heuristics
# ------------------------------------------------------------
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+.+$", re.MULTILINE)
ULIST = re.compile(r"^\s*[\-\*\+]\s+.+$", re.MULTILINE)
OLIST = re.compile(r"^\s*\d+\.\s+.+$", re.MULTILINE)
BLOCKQUOTE = re.compile(r"^\s{0,3}>\s+.+$", re.MULTILINE)
BACKTICK_BLOCK = re.compile(r"```")
INLINE_CODE = re.compile(r"`[^`\n]+`")
LINK_OR_IMG = re.compile(r"!?\[.*?\]\(.+?\)")
HR = re.compile(r"^(\s*\*\s*){3,}$|^(\s*\-{3,}\s*)$", re.MULTILINE)


def looks_like_markdown(text: str, threshold: int = 3) -> Tuple[bool, List[str]]:
    """
    Very fast heuristic that says:
    'Does *text* look like Markdown (vs plain text / HTML)?'

    Parameters
    ----------
    text : str
        The input to judge.
    threshold : int, optional
        Minimum number of positive indicators before we say “yes”.

    Returns
    -------
    Tuple[bool, List[str]]
        (flag, list_of_reasons)

    The list is handy for debugging / logging why we made the
    decision.
    """
    reasons = []

    # 1. ratio of MD-only line starters ----------------------------------------
    positive_count = 0

    for pattern, tag in [
        (HEADING, "heading"),
        (ULIST, "ul_list"),
        (OLIST, "ol_list"),
        (BLOCKQUOTE, "blockquote"),
        (HR, "horizontal_rule"),
    ]:
        if pattern.findall(text):
            positive_count += len(pattern.findall(text))
            reasons.append(tag)

    # 2. frequent inline / fenced constructs -----------------------------------
    if len(BACKTICK_BLOCK.findall(text)) >= 1:
        positive_count += 1
        reasons.append("code_fence")
    if len(INLINE_CODE.findall(text)) >= 3:
        positive_count += 1
        reasons.append("inline_code")
    if len(LINK_OR_IMG.findall(text)) >= 2:
        positive_count += 1
        reasons.append("link_or_img")

    # 3. Final verdict ---------------------------------------------------------
    return (positive_count >= threshold), reasons


# ----------------------------------------------------------------------
# Quick demo ------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        """# Hello world!

This is **bold** and this is `inline code`.
Here is a list:
- one
- two
""",
        """Hello world.\n\nThis is a plain text without formatting.\nThank you!""",
    ]

    for txt in samples:
        flag, why = looks_like_markdown(txt)
        print("Looks like Markdown?", flag)
        print("Reasons:", why)
        print("-" * 40)
