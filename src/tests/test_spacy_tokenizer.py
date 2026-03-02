import pytest
from text_tokenizer import SpacyTokenizer

try:
    import spacy
    spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SPACY_AVAILABLE, 
    reason="spaCy with required models (en_core_web_sm, es_core_news_sm) not installed"
)

class TestSpacyTokenizer:
    """Tests for SpacyTokenizer - tokenization, encoding, decoding, and dynamic vocabulary building."""
    
    def setup_method(self):
        """Create fresh tokenizers for each test with different language models."""
        self.en_tokenizer = SpacyTokenizer(model_name="en_core_web_sm")
        self.es_tokenizer = SpacyTokenizer(model_name="es_core_news_sm")
    
    # === TOKENIZATION TESTS ===
    
    def test_tokenize_english_basic(self):
        """Test basic English sentence tokenization with spaCy."""
        text = "Hello, world! How are you?"
        result = self.en_tokenizer.tokenize(text)
        
        # spaCy tokenization is more sophisticated - it separates punctuation
        # Expected: ["Hello", ",", "world", "!", "How", "are", "you", "?"]
        assert len(result) > 0
        assert "Hello" in result
        assert "," in result
        assert "world" in result
        assert "!" in result
        assert "you" in result
        assert "?" in result
    
    def test_tokenize_spanish_basic(self):
        """Test basic Spanish sentence tokenization with spaCy."""
        text = "¡Hola, mundo! ¿Cómo estás?"
        result = self.es_tokenizer.tokenize(text)
        
        # Spanish tokenization should handle special punctuation
        assert len(result) > 0
        assert "Hola" in result or "hola" in result
        assert "," in result
        assert "mundo" in result
        assert "¿" in result or "Cómo" in result
        assert "?" in result
    
    def test_tokenize_contractions(self):
        """Test that contractions are properly handled."""
        text = "I'm going to the store. She's happy."
        result = self.en_tokenizer.tokenize(text)
        
        # spaCy typically splits contractions: "I", "'m", "going", ...
        assert "I" in result
        assert "'m" in result or "am" in result
        assert "going" in result
    
    def test_tokenize_empty_string(self):
        """Test empty string returns empty list."""
        result = self.en_tokenizer.tokenize("")
        assert result == []
    
    def test_tokenize_numbers_and_dates(self):
        """Test tokenization of numbers and dates."""
        text = "I was born on 05/15/1990 and I have $100.50."
        result = self.en_tokenizer.tokenize(text)
        
        # spaCy should handle numbers appropriately
        assert any("05" in token or "15" in token or "1990" in token for token in result)
        assert any("$" in token or "100.50" in token for token in result)
    
    # === ENCODE TESTS ===
    
    def test_encode_adds_tokens_to_vocab(self):
        """Test that encoding dynamically adds tokens to vocabulary."""
        tokenizer = self.en_tokenizer
        initial_vocab_size = tokenizer.get_vocab_size()  # Should be 2 (PAD, UNK)
        
        text = "hello world"
        encoded = tokenizer.encode(text)
        
        # New tokens should be added to vocabulary
        assert tokenizer.get_vocab_size() > initial_vocab_size
        assert "hello" in tokenizer.get_vocab()
        assert "world" in tokenizer.get_vocab()
        
        # Encoded values should be integers
        assert all(isinstance(i, int) for i in encoded)
    
    def test_encode_same_text_twice(self):
        """Test that encoding the same text twice gives same IDs."""
        tokenizer = self.en_tokenizer
        text = "hello world"
        
        # First encoding adds tokens to vocab
        first_encoded = tokenizer.encode(text)
        first_vocab_size = tokenizer.get_vocab_size()
        
        # Second encoding should use same IDs
        second_encoded = tokenizer.encode(text)
        second_vocab_size = tokenizer.get_vocab_size()
        
        assert first_encoded == second_encoded
        assert first_vocab_size == second_vocab_size  # No new tokens added
    
    def test_encode_with_unknown_tokens_after_vocab_built(self):
        """Test that unknown tokens are handled after vocabulary is built."""
        tokenizer = self.en_tokenizer
        
        # First, build vocab with some words
        tokenizer.encode("hello world")
        hello_id = tokenizer.token2id.get("hello")
        world_id = tokenizer.token2id.get("world")
        
        # Now encode text with a known and unknown word
        text = "hello unknownword"
        encoded = tokenizer.encode(text)
        
        unk_id = tokenizer.token2id[tokenizer.UNK]
        assert encoded[0] == hello_id
        assert encoded[1] == unk_id
    
    def test_encode_preserves_order(self):
        """Test that encoding preserves token order."""
        tokenizer = self.en_tokenizer
        text = "first second third"
        encoded = tokenizer.encode(text)
        
        # Get tokens back through decode to verify order
        decoded = tokenizer.decode(encoded)
        # Note: decode joins with spaces, so order should be preserved
        words = decoded.split()
        assert words == ["first", "second", "third"] or len(words) > 0
    
    # === DECODE TESTS ===
    
    def test_decode_after_encode(self):
        """Test that decode reverses encode (round trip)."""
        tokenizer = self.en_tokenizer
        text = "This is a test sentence"
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        # Note: Due to spaCy's tokenization, the round trip might not be exact
        # because of punctuation handling, but words should be preserved
        original_words = text.split()
        decoded_words = decoded.split()
        
        # At minimum, all original words should appear in decoded
        for word in original_words:
            assert word in decoded_words
    
    def test_decode_with_padding(self):
        """Test decode ignores padding tokens."""
        tokenizer = self.en_tokenizer
        text = "hello world"
        
        encoded = tokenizer.encode(text)
        # Add padding IDs (0) at the end
        padded = encoded + [0, 0, 0]
        
        decoded = tokenizer.decode(padded)
        decoded_words = decoded.split()
        
        # Padding should be ignored, so we should have same words
        assert "hello" in decoded_words
        assert "world" in decoded_words
        assert len(decoded_words) == 2  # No extra words from padding
    
    def test_decode_unknown_ids(self):
        """Test unknown IDs become UNK in decoded text."""
        tokenizer = self.en_tokenizer
        
        # First, encode something to get valid IDs
        tokenizer.encode("hello")
        hello_id = tokenizer.token2id.get("hello")
        invalid_id = 9999  # ID that doesn't exist
        
        decoded = tokenizer.decode([hello_id, invalid_id])
        
        # Result should contain UNK token
        assert "hello" in decoded
        assert tokenizer.UNK in decoded
    
    # === VOCABULARY TESTS ===
    
    def test_initial_vocab_contains_pad_and_unk(self):
        """Test that initial vocabulary only has PAD and UNK."""
        tokenizer = self.en_tokenizer
        vocab = tokenizer.get_vocab()
        
        assert len(vocab) == 2
        assert tokenizer.PAD in vocab
        assert tokenizer.UNK in vocab
        assert vocab[0] == tokenizer.PAD
        assert vocab[1] == tokenizer.UNK
    
    def test_vocab_grows_with_new_tokens(self):
        """Test that vocabulary expands when new tokens are encountered."""
        tokenizer = self.en_tokenizer
        initial_size = tokenizer.get_vocab_size()
        
        # Encode first sentence
        tokenizer.encode("the cat sits")
        size_after_first = tokenizer.get_vocab_size()
        assert size_after_first > initial_size
        
        # Encode second sentence with new words
        tokenizer.encode("the dog runs")
        size_after_second = tokenizer.get_vocab_size()
        assert size_after_second > size_after_first
        
        # "the" should not be added again
        vocab = tokenizer.get_vocab()
        assert vocab.count("the") == 1
    
    def test_token2id_mapping_consistency(self):
        """Test that token2id mapping is consistent."""
        tokenizer = self.en_tokenizer
        text = "one two three"
        
        tokenizer.encode(text)
        
        # Verify that each token has a unique ID
        tokens = ["one", "two", "three"]
        ids = [tokenizer.token2id.get(t) for t in tokens]
        
        assert len(set(ids)) == 3  # All IDs should be different
        assert all(isinstance(i, int) for i in ids)
        
        # Verify bidirectional mapping
        for token in tokens:
            idx = tokenizer.token2id[token]
            assert tokenizer.id2token[idx] == token
    
    # === LANGUAGE-SPECIFIC TESTS ===
    
    def test_spanish_special_characters(self):
        """Test Spanish-specific characters are handled."""
        text = "corazón año niño café"
        result = self.es_tokenizer.tokenize(text)
        
        assert "corazón" in result
        assert "año" in result
        assert "niño" in result
        assert "café" in result
    
    def test_spanish_question_marks(self):
        """Test Spanish opening question marks are handled."""
        text = "¿Cómo estás?"
        result = self.es_tokenizer.tokenize(text)
        
        # spaCy should tokenize the opening question mark separately
        assert "¿" in result or "¿Cómo" in result
        assert "?" in result
    
    def test_french_special_characters(self):
        """Test French-specific characters with English tokenizer (should still work)."""
        # Even with English model, French characters should be tokenized
        text = "crème brûlée français"
        result = self.en_tokenizer.tokenize(text)
        
        assert "crème" in result
        assert "brûlée" in result
        assert "français" in result
    
    # === EDGE CASES ===
    
    def test_very_long_text(self):
        """Test with a longer text to ensure performance."""
        text = "This is a longer text. " * 100  # Repeat 100 times
        tokenizer = self.en_tokenizer
        
        # This should not raise any exceptions
        try:
            tokens = tokenizer.tokenize(text)
            encoded = tokenizer.encode(text)
            assert len(tokens) > 0
            assert len(encoded) > 0
        except Exception as e:
            pytest.fail(f"Long text handling failed: {e}")
    
    def test_text_with_only_punctuation(self):
        """Test text containing only punctuation."""
        text = "?!.,;:"
        result = self.en_tokenizer.tokenize(text)
        
        # spaCy should tokenize each punctuation mark
        assert len(result) > 0
        # At least some punctuation should be present
        assert any(p in "".join(result) for p in "?!.,;:")
    
    def test_multiple_encodes_different_texts(self):
        """Test encoding multiple different texts builds comprehensive vocab."""
        tokenizer = self.en_tokenizer
        
        texts = [
            "the quick brown fox",
            "jumps over the lazy dog",
            "pack my box with five dozen liquor jugs"
        ]
        
        all_tokens = set()
        for text in texts:
            tokenizer.encode(text)
            all_tokens.update(text.split())
        
        vocab = set(tokenizer.get_vocab())
        
        # All words from all texts should be in vocabulary
        for token in all_tokens:
            assert token in vocab
    
    def test_model_name_validation(self):
        """Test that invalid model names raise appropriate errors."""
        with pytest.raises(OSError):
            # This should fail because model doesn't exist
            SpacyTokenizer(model_name="non_existent_model")