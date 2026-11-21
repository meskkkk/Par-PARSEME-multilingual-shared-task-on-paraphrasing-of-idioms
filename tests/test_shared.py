"""Quick checks of the shared code that need no data, GPU or internet.
Run with:  python -m pytest -q tests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.dict_swap import swap  # noqa: E402
from src.shared.idiom_finder import locate  # noqa: E402
from src.shared.preprocess import clean_text, idiom_segments, mark_spans, project_marks  # noqa: E402


def test_clean_text_keeps_words():
    assert clean_text("  Le chat  dort​. ") == "Le chat dort."
    assert clean_text("École") == "École"  # no lowercasing


def test_project_marks_simple():
    tok = "les Nations Unies d' [[année internationale]] pour"
    raw = "les Nations Unies d'année internationale pour"
    assert project_marks(tok, raw) == "les Nations Unies d'[[année internationale]] pour"


def test_project_marks_contraction():
    tok = "Sem aéreo , inclui [[café de a manhã]] ."
    raw = "Sem aéreo, inclui café da manhã."
    assert idiom_segments(project_marks(tok, raw)) == ["café da manhã"]


def test_locate_discontinuous():
    s = "She took him to task."
    spans = locate(s, "took ... to task")
    assert mark_spans(s, spans) == "She [[took]] him [[to task]]."
    assert locate(s, "kicked the bucket") is None


def test_swap():
    assert swap("the managers were [[shown the door]].", "dismissed") == "the managers were dismissed."
    assert swap("She [[took]] him [[to task]].", "criticised") == "She criticised him."
    assert swap("[[Break a leg]] tonight", "good luck") == "Good luck tonight"
