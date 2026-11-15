# tests/affinity_pipeline/test_render.py
import pytest
from pathlib import Path
from affinity_pipeline.render import render_header, render_user_turn, finalize_answer, Transcript, TurnMeta

def test_render_header_none():
    """Test render_header with None system."""
    result = render_header(None)
    expected = "<|turn>system\n<|think|>\n<turn|>\n"
    assert result == expected

def test_render_header_empty_string():
    """Test render_header with empty string system."""
    result = render_header("")
    expected = "<|turn>system\n<|think|>\n<turn|>\n"
    assert result == expected

def test_render_header_non_empty():
    """Test render_header with non-empty string system."""
    result = render_header("Test system message")
    expected = "<|turn>system\n<|think|>\nTest system message<turn|>\n"
    assert result == expected

def test_render_user_turn_basic():
    """Test render_user_turn with basic text."""
    result = render_user_turn("Hello world!")
    expected = "<|turn>user\nHello world!<turn|>\n"
    assert result == expected

def test_render_user_turn_multibyte():
    """Test render_user_turn with multi-byte UTF-8 characters."""
    result = render_user_turn("你好世界 🌍")
    expected = "<|turn>user\n你好世界 🌍<turn|>\n"
    assert result == expected

def test_finalize_answer_no_turn_close():
    """Test finalize_answer when answer doesn't end with <turn|>."""
    result = finalize_answer("Some text")
    expected = "Some text<turn|>"
    assert result == expected

def test_finalize_answer_has_turn_close():
    """Test finalize_answer when answer already ends with <turn|>."""
    result = finalize_answer("Some text<turn|>")
    expected = "Some text<turn|>"
    assert result == expected

def test_finalize_answer_has_turn_close_with_whitespace():
    """Test finalize_answer when answer ends with <turn|> and whitespace."""
    result = finalize_answer("Some text<turn|>   ")
    expected = "Some text<turn|>   "
    assert result == expected

def test_transcript_open_user_turn_ascii():
    """Test Transcript.open_user_turn with ASCII characters."""
    transcript = Transcript(None)
    user_start, user_end, model_start = transcript.open_user_turn("Hello")

    # Check positions (absolute in body)
    # header (32) + user_turn (25) + model_open (13)
    assert user_start == 32
    assert user_end == 57  # 32 + len("<|turn>user\nHello<turn|>\n")
    assert model_start == 70  # 57 + len("<|turn>model\n")

    # Check body content
    expected_body = "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n"
    assert transcript.body == expected_body

def test_transcript_open_user_turn_multibyte():
    """Test Transcript.open_user_turn with multi-byte characters."""
    transcript = Transcript(None)
    user_start, user_end, model_start = transcript.open_user_turn("你好")

    # Check positions (absolute in body, multi-byte chars count as single code points in Python str)
    # header (32) + user_turn (22: 12 + 2 + 8) + model_open (13)
    assert user_start == 32
    assert user_end == 54  # 32 + len("<|turn>user\n你好<turn|>\n") = 32 + 22
    assert model_start == 67  # 54 + 13

    # Check body content
    expected_body = "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n你好<turn|>\n<|turn>model\n"
    assert transcript.body == expected_body

def test_transcript_open_user_turn_cumulative():
    """Test Transcript.open_user_turn with multiple calls."""
    transcript = Transcript(None)

    # First turn
    # header (32) + user_turn (25: 12+5+8)
    user_start1, user_end1, model_start1 = transcript.open_user_turn("First")
    assert user_start1 == 32
    assert user_end1 == 57  # 32 + 25
    assert model_start1 == 70  # 57 + 13

    # Second turn
    # body now: header(32) + first_user(25) + model_open(13) = 70
    # second_user(26: 12+6+8)
    user_start2, user_end2, model_start2 = transcript.open_user_turn("Second")
    assert user_start2 == 70  # After first model open
    assert user_end2 == 96  # 70 + 26
    assert model_start2 == 109  # 96 + 13

    expected_body = (
        "<|turn>system\n<|think|>\n<turn|>\n"
        "<|turn>user\nFirst<turn|>\n<|turn>model\n"
        "<|turn>user\nSecond<turn|>\n<|turn>model\n"
    )
    assert transcript.body == expected_body

def test_transcript_close_model_turn():
    """Test Transcript.close_model_turn."""
    transcript = Transcript(None)
    user_start, user_end, model_start = transcript.open_user_turn("User message")

    # Close the turn
    meta = transcript.close_model_turn(
        turn=1,
        label="test",
        user_start=user_start,
        user_end=user_end,
        model_start=model_start,
        answer="Model response",
        stop_type="eos",
        tokens_predicted=50,
        seed_used=42,
        gen_tokens=[1, 2, 3]
    )

    # Check body
    expected_body = (
        "<|turn>system\n<|think|>\n<turn|>\n"
        "<|turn>user\nUser message<turn|>\n<|turn>model\n"
        "Model response<turn|>\n"
    )
    assert transcript.body == expected_body

    # Check meta (positions passed in from open_user_turn)
    assert meta.turn == 1
    assert meta.label == "test"
    assert meta.user_start == 32
    assert meta.user_end == 64  # header(32) + user_turn(32)
    assert meta.model_start == 77  # 64 + model_open(13)
    assert meta.model_end == 98  # 77 + finalized_answer(21)
    assert meta.stop_type == "eos"
    assert meta.tokens_predicted == 50
    assert meta.seed_used == 42
    assert meta.gen_tokens == [1, 2, 3]

    # Check that turn was added to transcript
    assert len(transcript.turns) == 1
    assert transcript.turns[0] is meta

def test_transcript_rollback():
    """Test Transcript.rollback."""
    transcript = Transcript(None)
    user_start, user_end, model_start = transcript.open_user_turn("User message")

    # Add a turn
    meta = transcript.close_model_turn(
        turn=1,
        label="test",
        user_start=user_start,
        user_end=user_end,
        model_start=model_start,
        answer="Model response",
        stop_type="eos",
        tokens_predicted=50,
        seed_used=42,
        gen_tokens=None
    )

    # Verify body has the turn (header + user_turn + model_open + response + newline)
    assert len(transcript.body) > 0

    # Rollback to user_start (position before user turn was added)
    transcript.rollback(user_start)

    # Body should be restored to just the header (before the turn)
    expected_body = "<|turn>system\n<|think|>\n<turn|>\n"
    assert transcript.body == expected_body

def test_turnmeta_to_dict():
    """Test TurnMeta.to_dict key order and values."""
    meta = TurnMeta(
        turn=1,
        label="test",
        user_start=0,
        user_end=10,
        model_start=11,
        model_end=20,
        stop_type="eos",
        tokens_predicted=50,
        seed_used=42,
        gen_tokens=[1, 2, 3]
    )
    
    result = meta.to_dict()
    
    # Check key order
    keys = list(result.keys())
    expected_keys = [
        "turn", "label", "user_start", "user_end", "model_start", 
        "model_end", "stop_type", "tokens_predicted", "seed_used", "gen_tokens"
    ]
    assert keys == expected_keys
    
    # Check values
    assert result["turn"] == 1
    assert result["label"] == "test"
    assert result["user_start"] == 0
    assert result["user_end"] == 10
    assert result["model_start"] == 11
    assert result["model_end"] == 20
    assert result["stop_type"] == "eos"
    assert result["tokens_predicted"] == 50
    assert result["seed_used"] == 42
    assert result["gen_tokens"] == [1, 2, 3]

def test_turnmeta_to_dict_none_tokens():
    """Test TurnMeta.to_dict with None gen_tokens."""
    meta = TurnMeta(
        turn=1,
        label="test",
        user_start=0,
        user_end=10,
        model_start=11,
        model_end=20,
        stop_type="eos",
        tokens_predicted=50,
        seed_used=42,
        gen_tokens=None
    )
    
    result = meta.to_dict()
    
    # Check that gen_tokens is None
    assert result["gen_tokens"] is None

def test_transcript_close_model_turn_positioning():
    """Test Transcript.close_model_turn model_end positioning."""
    transcript = Transcript(None)
    user_start, user_end, model_start = transcript.open_user_turn("User")

    # Close the turn with a response that has no trailing <turn|>
    meta = transcript.close_model_turn(
        turn=1,
        label="test",
        user_start=user_start,
        user_end=user_end,
        model_start=model_start,
        answer="Response",  # No trailing <turn|> in answer
        stop_type="eos",
        tokens_predicted=5,
        seed_used=42,
        gen_tokens=None
    )

    # Check that model_end is positioned before the newline (after final <turn|>)
    # The body should be:
    # "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nUser<turn|>\n<|turn>model\nResponse<turn|>\n"
    # Calculation: header(32) + user_turn(24) + model_open(13) + finalized_answer(15) = 84
    expected_body = (
        "<|turn>system\n<|think|>\n<turn|>\n"
        "<|turn>user\nUser<turn|>\n<|turn>model\n"
        "Response<turn|>\n"
    )

    assert transcript.body == expected_body
    assert meta.model_end == 84  # Position after final <turn|>, before newline

def test_transcript_multiple_turns():
    """Test multiple turns with proper positioning."""
    transcript = Transcript("System message")
    
    # First turn
    user_start1, user_end1, model_start1 = transcript.open_user_turn("First user")
    
    meta1 = transcript.close_model_turn(
        turn=1,
        label="first",
        user_start=user_start1,
        user_end=user_end1,
        model_start=model_start1,
        answer="First response",
        stop_type="eos",
        tokens_predicted=5,
        seed_used=42,
        gen_tokens=None
    )
    
    # Second turn
    user_start2, user_end2, model_start2 = transcript.open_user_turn("Second user")
    
    meta2 = transcript.close_model_turn(
        turn=2,
        label="second",
        user_start=user_start2,
        user_end=user_end2,
        model_start=model_start2,
        answer="Second response",
        stop_type="eos",
        tokens_predicted=5,
        seed_used=43,
        gen_tokens=None
    )
    
    # Verify final body has both turns
    expected_body = (
        "<|turn>system\n<|think|>\nSystem message<turn|>\n"
        "<|turn>user\nFirst user<turn|>\n<|turn>model\n"
        "First response<turn|>\n"
        "<|turn>user\nSecond user<turn|>\n<|turn>model\n"
        "Second response<turn|>\n"
    )
    
    assert transcript.body == expected_body
    
    # Verify both turns in list
    assert len(transcript.turns) == 2
    assert transcript.turns[0] is meta1
    assert transcript.turns[1] is meta2

def test_finalize_answer_edge_cases():
    """Test finalize_answer with edge cases."""
    # Empty string
    assert finalize_answer("") == "<turn|>"

    # Only whitespace (rstrip removes it, then add <turn|>)
    assert finalize_answer("   ") == "<turn|>"

    # Whitespace + <turn|> (rstrip preserves <turn|>, returns original)
    assert finalize_answer("  <turn|>") == "  <turn|>"

    # <turn|> at end with newline
    assert finalize_answer("text\n<turn|>\n") == "text\n<turn|>\n"

    # Multiple <turn|> at the end
    assert finalize_answer("text<turn|><turn|>") == "text<turn|><turn|>"

def test_render_user_turn_special_chars():
    """Test render_user_turn with special characters."""
    result = render_user_turn("User: \"Quote\" and 'apostrophe'")
    expected = "<|turn>user\nUser: \"Quote\" and 'apostrophe'<turn|>\n"
    assert result == expected
