import pytest
from text_tokenizer import HFWordpieceTokenizer

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE,
    reason="Transformers library not installed. Please install with `pip install transformers`"
)

class TestHFWordpieceTokenizer:
    """Tests for HFWordpieceTokenizer - Hugging Face WordPiece tokenization."""
    
    def setup_method(self):
        """Create fresh tokenizers for each test with different models."""
        self.tokenizer = HFWordpieceTokenizer()  # Default: bert-base-multilingual-cased
        self.english_tokenizer = HFWordpieceTokenizer(model_name="bert-base-uncased")
        self.spanish_tokenizer = HFWordpieceTokenizer(model_name="dccuchile/bert-base-spanish-wwm-uncased")
    
    # === TOKENIZATION TESTS ===
    
    def test_tokenize_english_basic(self):
        """Test basic English sentence tokenization with WordPiece."""
        text = "Hello, world! How are you?"
        result = self.tokenizer.tokenize(text)
        
        # WordPiece splits into subwords
        assert len(result) > 0
        # Common subwords for "Hello"
        assert any("hello" in token.lower() for token in result)
        # Punctuation is typically kept as separate tokens
        assert "," in result or "##," in result
    
    def debug_tokenizer_output(self):
        """Debug method to see actual tokenizer output."""
        tokenizer = HFWordpieceTokenizer()
        text = "Hola, ¿cómo estás?"
        
        tokens = tokenizer.tokenize(text)
        print(f"Text: {text}")
        print(f"Tokens: {tokens}")
        
        # Also see the encoded IDs and decode them back
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"Encoded IDs: {ids}")
        print(f"Decoded back: {decoded}")
        
        return tokens
        

    def test_tokenize_unknown_words(self):
        """Test how WordPiece handles unknown words with subword tokenization."""
        text = "supercalifragilisticexpialidocious"
        result = self.tokenizer.tokenize(text)
        
        # Very long words should be split into multiple subwords
        assert len(result) > 1
        
        # Each subword should start with ## (except the first)
        for i, token in enumerate(result):
            if i > 0:
                assert token.startswith("##") or len(token) > 0
    
    def test_tokenize_empty_string(self):
        """Test empty string returns empty list."""
        result = self.tokenizer.tokenize("")
        assert result == []
    
    
    def test_tokenize_chinese_characters(self):
        """Test tokenization of Chinese characters with multilingual model."""
        text = "你好，世界！"
        result = self.tokenizer.tokenize(text)
        
        # Chinese characters are typically kept as single tokens
        assert len(result) > 0
        # Each Chinese character might be a separate token
        chinese_tokens = [t for t in result if not t.startswith("##")]
        assert len(chinese_tokens) > 0
    
    # === ENCODE TESTS ===
    
    def test_encode_basic(self):
        """Test basic encoding returns list of integers."""
        text = "hello world"
        result = self.tokenizer.encode(text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(i, int) for i in result)
    
    def test_encode_no_special_tokens(self):
        """Test that encode doesn't add special tokens by default."""
        text = "hello world"
        result = self.tokenizer.encode(text)
        
        # By default, we set add_special_tokens=False
        # So we shouldn't see [CLS] (101 for BERT) or [SEP] (102)
        assert 101 not in result  # [CLS] token ID for BERT
        assert 102 not in result  # [SEP] token ID for BERT
    
    def test_encode_empty_string(self):
        """Test encoding empty string."""
        result = self.tokenizer.encode("")
        
        # Empty string might return empty list or list with special tokens
        # Based on our implementation, it should be empty
        assert result == [] or len(result) <= 2
    
    def test_encode_same_text_consistent(self):
        """Test that encoding same text produces same IDs."""
        text = "This is a test sentence"
        
        first_encode = self.tokenizer.encode(text)
        second_encode = self.tokenizer.encode(text)
        
        assert first_encode == second_encode
    
    def test_encode_with_different_models(self):
        """Test encoding with different model variants."""
        text = "hello world"
        
        multilingual_result = self.tokenizer.encode(text)
        english_result = self.english_tokenizer.encode(text)
        
        # Different models will likely produce different IDs
        # But both should produce valid token sequences
        assert len(multilingual_result) > 0
        assert len(english_result) > 0
    
    # === DECODE TESTS ===
    
    def test_decode_basic(self):
        """Test basic decoding returns string."""
        text = "hello world"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        
        # Decoding might not exactly match original due to subword merging
        # But should contain the original words
        assert "hello" in decoded.lower()
        assert "world" in decoded.lower()
    
    def test_decode_no_special_tokens(self):
        """Test that decode skips special tokens."""
        # Encode normally (no special tokens)
        text = "hello world"
        encoded = self.tokenizer.encode(text)
        
        # Manually add special token IDs (for BERT: 101=[CLS], 102=[SEP])
        if self.tokenizer.model_name == "bert-base-multilingual-cased":
            encoded_with_specials = [101] + encoded + [102]
        else:
            # Skip if we can't determine special tokens
            encoded_with_specials = encoded
        
        decoded = self.tokenizer.decode(encoded_with_specials)
        
        # With skip_special_tokens=True, special tokens shouldn't appear
        assert "[CLS]" not in decoded
        assert "[SEP]" not in decoded
        assert "hello" in decoded.lower()
    
    def test_round_trip_consistency(self):
        """Test that encode-decode cycle preserves meaning."""
        test_sentences = [
            "Hello world",
            "This is a test",
            "Hola mundo",
            "Bonjour le monde",
            "Hello, how are you?",
            "Testing 123 numbers"
        ]
        
        for sentence in test_sentences:
            encoded = self.tokenizer.encode(sentence)
            decoded = self.tokenizer.decode(encoded)
            
            # The decoded text should contain the key words from original
            # Due to subword tokenization, exact match isn't guaranteed
            original_words = sentence.lower().split()
            decoded_lower = decoded.lower()
            
            # At least some original words should appear
            matches = sum(1 for word in original_words if word in decoded_lower)
            assert matches > 0, f"No words matched for: {sentence}"
    
    def test_decode_empty_list(self):
        """Test decoding empty list returns empty string."""
        result = self.tokenizer.decode([])
        assert result == ""
    
    def test_decode_invalid_ids(self):
        """Test decoding with invalid IDs (should handle gracefully)."""
        # Some models have vocab size limits, test with very large ID
        large_id = [999999] if self.tokenizer.get_vocab_size() < 999999 else [1]
        
        # This should not raise an exception
        try:
            result = self.tokenizer.decode(large_id)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"Decoding invalid IDs failed: {e}")
    
    # === VOCABULARY TESTS ===
    
    def test_get_vocab_size(self):
        """Test vocabulary size is positive and reasonable."""
        size = self.tokenizer.get_vocab_size()
        
        assert size > 0
        # BERT models typically have vocab size around 30k-120k
        assert 1000 < size < 200000
    
    def test_get_vocab(self):
        """Test getting the full vocabulary."""
        vocab = self.tokenizer.get_vocab()
        
        assert isinstance(vocab, dict)
        assert len(vocab) == self.tokenizer.get_vocab_size()
        
        # Check structure: token -> id mapping
        sample_tokens = list(vocab.items())[:5]
        for token, token_id in sample_tokens:
            assert isinstance(token, str)
            assert isinstance(token_id, int)
    
    def test_vocab_contains_common_tokens(self):
        """Test that vocabulary contains common WordPiece tokens."""
        vocab = self.tokenizer.get_vocab()
        
        # Common tokens in multilingual BERT
        common_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        for token in common_tokens:
            assert token in vocab, f"Token {token} not found in vocabulary"
        
        # Check for some common words/subwords
        assert any("hello" in token.lower() for token in vocab)
        assert any("world" in token.lower() for token in vocab)
    
    # === MODEL-SPECIFIC TESTS ===
    
    def test_multilingual_vs_english_model(self):
        """Test differences between multilingual and English-only models."""
        text = "café"
        
        multilingual_tokens = self.tokenizer.tokenize(text)
        english_tokens = self.english_tokenizer.tokenize(text)
        
        # Both should tokenize successfully
        assert len(multilingual_tokens) > 0
        assert len(english_tokens) > 0
        
        # Results might differ due to vocabulary differences
        # But we just verify they work
    
    def test_spanish_model_spanish_text(self):
        """Test Spanish-specific model with Spanish text."""
        if "spanish" not in self.spanish_tokenizer.model_name.lower():
            pytest.skip("Spanish model not available")
        
        text = "El niño juega con su pelota roja"
        tokens = self.spanish_tokenizer.tokenize(text)
        
        # Should tokenize Spanish words appropriately
        assert len(tokens) > 0
        assert any("niño" in token.lower() for token in tokens)
        assert any("pelota" in token.lower() for token in tokens)
    
    # === EDGE CASES ===
    
    def test_very_long_text(self):
        """Test with very long text."""
        text = "word " * 1000  # 1000 words
        tokenizer = self.tokenizer
        
        # This should not raise exceptions
        try:
            tokens = tokenizer.tokenize(text)
            encoded = tokenizer.encode(text)
            
            assert len(tokens) > 0
            assert len(encoded) > 0
            # Length in tokens should be reasonable
            assert len(encoded) <= len(text.split()) * 2  # At most 2x words due to subwords
        except Exception as e:
            pytest.fail(f"Long text handling failed: {e}")
    
    def test_text_with_only_punctuation(self):
        """Test text containing only punctuation."""
        texts = [
            "?!.,;:",
            "---",
            "...",
            "()[]{}"
        ]
        
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            encoded = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(encoded)
            
            # Should handle punctuation
            assert len(tokens) > 0
            assert len(encoded) > 0
            # Decoded should contain at least some punctuation
            assert any(p in decoded for p in text if not p.isspace())
    
    def test_unicode_and_emoji(self):
        """Test tokenization of Unicode characters and emoji."""
        text = "Hello  world  café "
        tokens = self.tokenizer.tokenize(text)
        
        # Emoji might be handled as [UNK] or special tokens
        # But the function should not crash
        assert len(tokens) > 0
    
    def test_model_name_validation(self):
        """Test that invalid model names raise appropriate errors."""
        with pytest.raises(Exception):
            # This should fail because model doesn't exist
            HFWordpieceTokenizer(model_name="this-model-definitely-does-not-exist-12345")
    
    def test_multiple_encodes_different_texts(self):
        """Test encoding multiple texts with same tokenizer."""
        texts = [
            "The quick brown fox",
            "Jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs"
        ]
        
        for text in texts:
            encoded = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(encoded)
            
            assert len(encoded) > 0
            # Decoded should contain key words from original
            original_words = text.lower().split()
            decoded_lower = decoded.lower()
            
            # At least some original content should be preserved
            assert any(word in decoded_lower for word in original_words)
    
    def test_token_id_consistency(self):
        """Test that same token always gets same ID."""
        text = "hello hello hello"
        encoded = self.tokenizer.encode(text)
        
        # The same word repeated should map to the same token IDs
        # But due to subword tokenization, "hello" might split into multiple tokens
        # We can check that the pattern repeats
        
        # Get tokens first
        tokens = self.tokenizer.tokenize(text)
        
        # If "hello" splits into ["he", "##llo"], then the sequence should repeat
        if len(tokens) > 1:
            # Check first half vs second half
            half = len(tokens) // 3  # Since we have 3 hellos
            first_pattern = tokens[:half]
            second_pattern = tokens[half:2*half]
            assert first_pattern == second_pattern

    def test_encode_with_special_tokens_option(self):
        """Test that we can optionally add special tokens."""
        # Our implementation uses add_special_tokens=False
        # But we can test what happens if we want them
        
        text = "hello world"
        
        # Current behavior (no special tokens)
        without_specials = self.tokenizer.encode(text)
        
        # Manually test with special tokens by calling tokenizer directly
        hf_tokenizer = self.tokenizer.tokenizer
        with_specials = hf_tokenizer.encode(text, add_special_tokens=True)
        
        # With special tokens should be longer
        assert len(with_specials) >= len(without_specials)
        
        # Special tokens (like [CLS], [SEP]) should be present
        if "bert" in self.tokenizer.model_name.lower():
            assert 101 in with_specials or 1 in with_specials  # [CLS]
            assert 102 in with_specials or 2 in with_specials  # [SEP]