import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.filter_research_english import filter_reasons


def test_requested_foreign_output_excluded_even_in_latin_script():
    reasons=filter_reasons({'question':'Translate the following English sentence into French: Hello world.','answer':'Bonjour le monde.'})
    assert any(r['reason']=='foreign_output_instruction' and r['requested_language']=='french' for r in reasons)


def test_translation_into_english_can_be_retained():
    assert not filter_reasons({'question':'Translate the following French sentence into English: Bonjour.','answer':'Hello.'})


def test_actual_non_latin_answer_does_not_inherit_english_source_label():
    reasons=filter_reasons({'question':'Summarize this text in English.','answer':'В одном из районов Алматы была подана официальная жалоба.'})
    assert any(r['reason']=='non_latin_script_screen' for r in reasons)


def test_explicit_foreign_reply_instruction_is_excluded():
    reasons=filter_reasons({'question':'Please reply to this question in Spanish: How are you?','answer':'Muy bien.'})
    assert any(r['reason']=='foreign_output_instruction' for r in reasons)


def test_image_ocr_question_is_not_a_foreign_output_instruction():
    assert not filter_reasons({'question':'What does it say in french?\nYour answer should be very brief.','answer':'Bonjour.'})
