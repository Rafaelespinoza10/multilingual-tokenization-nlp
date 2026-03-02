import pytest
from text_tokenizer import EdgegramTokenizer

class TestEdgegramTokenizer:
    """Tests for EdgegramTokenizer - edge n-gram based tokenization."""
    
    def setup_method(self):
        """Create fresh tokenizers with different configurations for each test."""
        self.default_tokenizer = EdgegramTokenizer()
    
    # === ENCODE TESTS ===
    
    def test_encode_basic(self):
        """Test encoding converts tokens to IDs."""
        tokenizer = EdgegramTokenizer()
        text = "hello world"
        
        encoded = tokenizer.encode(text)
        
        # Should return list of integers
        assert isinstance(encoded, list)
        assert all(isinstance(i, int) for i in encoded)
        
        # Should have 4 tokens: "hel", "llo", "wor", "rld"
        assert len(encoded) == 4
        
        # First token should be "hel" (ID >= 2, since 0=PAD, 1=UNK)
        assert encoded[0] >= 2
    
    def test_encode_add_to_vocab_true_default(self):
        """Test that encode with default add_to_vocab=True adds new tokens."""
        tokenizer = EdgegramTokenizer()
        
        # Reset first to ensure clean state
        tokenizer.reset_vocabulary()
        assert tokenizer.get_vocab_size() == 2
        
        # Encode with default add_to_vocab=True
        encoded = tokenizer.encode("hello")
        
        # Should return valid IDs (not all UNK)
        unk_id = tokenizer.token2id[tokenizer.UNK]
        assert not all(id == unk_id for id in encoded)
        
        # Vocabulary should have grown
        assert tokenizer.get_vocab_size() > 2
        assert "hel" in tokenizer.get_vocab()
        assert "llo" in tokenizer.get_vocab()
        
        # The encoded IDs should match the token IDs
        assert tokenizer.token2id["hel"] == encoded[0]
        assert tokenizer.token2id["llo"] == encoded[1]
    
    def test_encode_add_to_vocab_false(self):
        """Test that encode with add_to_vocab=False doesn't add new tokens."""
        tokenizer = EdgegramTokenizer()
        
        # Reset first
        tokenizer.reset_vocabulary()
        assert tokenizer.get_vocab_size() == 2
        
        # Encode with add_to_vocab=False
        encoded = tokenizer.encode("hello", add_to_vocab=False)
        unk_id = tokenizer.token2id[tokenizer.UNK]
        
        # All tokens should be UNK
        assert all(id == unk_id for id in encoded)
        assert len(encoded) == 2  # "hel" and "llo" both become UNK
        
        # Vocabulary should NOT have grown
        assert tokenizer.get_vocab_size() == 2
        assert "hel" not in tokenizer.get_vocab()
        assert "llo" not in tokenizer.get_vocab()
    
    def test_encode_adds_to_vocabulary(self):
        """Test that encoding adds new tokens to vocabulary."""
        tokenizer = EdgegramTokenizer()
        initial_vocab_size = tokenizer.get_vocab_size()  # Should be 2 (PAD, UNK)
        
        tokenizer.encode("hello")
        
        # Vocabulary should have grown
        assert tokenizer.get_vocab_size() > initial_vocab_size
        
        # Should contain the edgegrams
        vocab = tokenizer.get_vocab()
        assert "hel" in vocab
        assert "llo" in vocab
    
    def test_encode_unknown_token(self):
        """Test that unknown tokens (after reset) become UNK when add_to_vocab=False."""
        tokenizer = EdgegramTokenizer()
        
        # First encode to build vocab (add_to_vocab=True by default)
        first_encoded = tokenizer.encode("hello")
        print(f"\nFirst encode result: {first_encoded}")
        print(f"Vocabulary after first encode: {tokenizer.get_vocab()}")
        
        # Verify tokens were added
        assert "hel" in tokenizer.get_vocab()
        assert "llo" in tokenizer.get_vocab()
        
        # Reset vocabulary
        tokenizer.reset_vocabulary()
        print(f"\nAfter reset - Vocabulary: {tokenizer.get_vocab()}")
        assert tokenizer.get_vocab_size() == 2  # Only PAD and UNK
        
        # Now encode again with add_to_vocab=False - tokens should become UNK
        second_encoded = tokenizer.encode("hello", add_to_vocab=False)
        print(f"Second encode result (add_to_vocab=False): {second_encoded}")
        
        unk_id = tokenizer.token2id[tokenizer.UNK]
        print(f"UNK ID: {unk_id}")
        
        # All tokens should be UNK
        for i, id_val in enumerate(second_encoded):
            print(f"  Token {i}: {id_val} == {unk_id}? {id_val == unk_id}")
            assert id_val == unk_id, f"Token {i} is {id_val}, expected {unk_id}"
        
        assert all(id == unk_id for id in second_encoded)
        
        # Verify vocabulary still only has PAD and UNK
        assert tokenizer.get_vocab_size() == 2
        assert "hel" not in tokenizer.get_vocab()
        assert "llo" not in tokenizer.get_vocab()

    def test_encode_empty_string(self):
        """Test encoding empty string returns empty list."""
        tokenizer = EdgegramTokenizer()
        assert tokenizer.encode("") == []
        assert tokenizer.encode("   ") == []
    
    # === DECODE TESTS ===
    
    def test_decode_basic(self):
        """Test decoding converts IDs back to tokens."""
        tokenizer = EdgegramTokenizer()
        text = "hello world"
        
        # First encode
        encoded = tokenizer.encode(text)
        
        # Then decode
        decoded = tokenizer.decode(encoded)
        
        # Should get back the tokens as a string
        assert isinstance(decoded, str)
        
        # Should contain all edgegrams
        assert "hel" in decoded
        assert "llo" in decoded
        assert "wor" in decoded
        assert "rld" in decoded
    
    def test_decode_with_padding(self):
        """Test that decode ignores padding tokens."""
        tokenizer = EdgegramTokenizer()
        
        # Encode some text
        encoded = tokenizer.encode("hello")
        
        # Add padding IDs (0) at the end
        padded = encoded + [0, 0, 0]
        
        # Decode should ignore padding
        decoded = tokenizer.decode(padded)
        decoded_tokens = decoded.split()
        
        # Should only have the original tokens
        assert len(decoded_tokens) == len(encoded)
        assert all(t in ["hel", "llo"] for t in decoded_tokens)
    
    def test_decode_unknown_ids(self):
        """Test decoding with unknown IDs becomes UNK."""
        tokenizer = EdgegramTokenizer()
        
        # Valid ID for "hel"
        tokenizer.encode("hello")
        valid_id = tokenizer.token2id["hel"]
        
        # Invalid ID
        invalid_id = 9999
        
        decoded = tokenizer.decode([valid_id, invalid_id])
        decoded_tokens = decoded.split()
        
        # First token should be "hel", second should be UNK
        assert decoded_tokens[0] == "hel"
        assert tokenizer.UNK in decoded_tokens[1]
    
    # === ROUND TRIP TESTS ===
    
    def test_round_trip_preserves_tokens(self):
        """Test that encode + decode preserves the original tokens."""
        tokenizer = EdgegramTokenizer()
        test_cases = [
            "hello",
            "hello world",
            "a b c",  # Single chars
            "cat dog elephant",  # Mixed lengths
            "Hello World",  # Capitalization preserved
        ]
        
        for text in test_cases:
            # Get tokens directly
            original_tokens = tokenizer.tokenize(text)
            
            # Encode and decode
            encoded = tokenizer.encode(text)
            decoded_tokens = tokenizer.decode(encoded).split()
            
            # Should have same tokens
            assert original_tokens == decoded_tokens, f"Failed for: '{text}'"
    
    def test_round_trip_with_repeated_words(self):
        """Test round trip with repeated words."""
        tokenizer = EdgegramTokenizer()
        text = "hello hello hello"
        
        tokens = tokenizer.tokenize(text)
        encoded = tokenizer.encode(text)
        decoded_tokens = tokenizer.decode(encoded).split()
        
        assert tokens == decoded_tokens
        assert len(decoded_tokens) == 6  # 3 words × 2 edgegrams each
    
    # === VOCABULARY TESTS ===
    
    def test_vocab_contains_pad_and_unk(self):
        """Test that vocabulary always has PAD and UNK."""
        tokenizer = EdgegramTokenizer()
        vocab = tokenizer.get_vocab()
        
        assert tokenizer.PAD in vocab
        assert tokenizer.UNK in vocab
        assert vocab[0] == tokenizer.PAD
        assert vocab[1] == tokenizer.UNK
    
    def test_vocab_size_updates(self):
        """Test that vocabulary size updates correctly."""
        tokenizer = EdgegramTokenizer()
        
        # Start with 2 (PAD, UNK)
        assert tokenizer.get_vocab_size() == 2
        
        # Add some tokens
        tokenizer.encode("hello world")
        size_after_first = tokenizer.get_vocab_size()
        assert size_after_first > 2
        
        # Add more tokens
        tokenizer.encode("goodbye friends")
        size_after_second = tokenizer.get_vocab_size()
        assert size_after_second > size_after_first
    
    def test_token2id_mapping(self):
        """Test token to ID mapping is consistent."""
        tokenizer = EdgegramTokenizer()
        tokenizer.encode("hello")
        
        token2id = tokenizer.get_token2id()
        
        # Check special tokens
        assert token2id[tokenizer.PAD] == 0
        assert token2id[tokenizer.UNK] == 1
        
        # Check generated tokens
        assert "hel" in token2id
        assert "llo" in token2id
        assert token2id["hel"] != token2id["llo"]
    
    def test_id2token_mapping(self):
        """Test ID to token mapping is consistent."""
        tokenizer = EdgegramTokenizer()
        tokenizer.encode("hello")
        
        id2token = tokenizer.get_id2token()
        
        # Check bidirectional mapping
        for token, id in tokenizer.token2id.items():
            assert id2token[id] == token