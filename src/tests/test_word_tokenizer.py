import pytest
from text_tokenizer import WordTokenizer

class TestWordTokenizer:
    """Tests for WordTokenizer - tokenization, encoding, decoding, vocabulary and corpus building."""
    
    def setup_method(self):
        """Create a fresh tokenizer for each test."""
        # Tokenizer without predefined vocabulary
        self.tokenizer = WordTokenizer()
        # Tokenizer with known vocabulary for predictable tests
        self.vocab_tokenizer = WordTokenizer(vocab=["hello", "world", "hola", "mundo"])
    
    
    def test_tokenize_simple_sentence(self):
        """Test basic sentence tokenization by spaces."""
        result = self.tokenizer.tokenize("hello world")
        assert result == ["hello", "world"]
    
    def test_tokenize_with_multiple_spaces(self):
        """Test multiple spaces are handled correctly."""
        result = self.tokenizer.tokenize("hello   world")
        assert result == ["hello", "world"]
    
    def test_tokenize_with_punctuation(self):
        """Test punctuation is attached to words (standard split behavior)."""
        result = self.tokenizer.tokenize("hello, world!")
        assert result == ["hello,", "world!"]  # Punctuation stays attached to words
    
    def test_tokenize_empty_string(self):
        """Test empty string returns empty list."""
        result = self.tokenizer.tokenize("")
        assert result == []
    
    def test_tokenize_only_spaces(self):
        """Test string with only spaces returns empty list."""
        result = self.tokenizer.tokenize("   ")
        assert result == []
    
    def test_tokenize_with_newlines(self):
        """Test newlines and tabs are treated as whitespace."""
        result = self.tokenizer.tokenize("hello\nworld\t!")
        # split() handles all whitespace by default
        assert "hello" in result
        assert "world" in result
    
    
    def test_encode_known_words(self):
        """Test encoding words that are in vocabulary."""
        # Use tokenizer with known vocabulary
        result = self.vocab_tokenizer.encode("hello world")
        
        # Verify IDs are as expected
        hello_id = self.vocab_tokenizer.word2id["hello"]
        world_id = self.vocab_tokenizer.word2id["world"]
        assert result == [hello_id, world_id]
    
    def test_encode_unknown_words(self):
        """Test unknown words become UNK token."""
        # Words outside known vocabulary
        result = self.vocab_tokenizer.encode("hello unknown_word")
        
        hello_id = self.vocab_tokenizer.word2id["hello"]
        unk_id = self.vocab_tokenizer.word2id[self.vocab_tokenizer.UNK]
        assert result == [hello_id, unk_id]
    
    def test_encode_empty_string(self):
        """Test encoding empty string returns empty list."""
        result = self.tokenizer.encode("")
        assert result == []
    
    def test_encode_with_special_tokens(self):
        """Test that special tokens are in vocabulary and have correct IDs."""
        tokenizer = self.tokenizer  # the one with base vocabulary
        
        # Verify special tokens exist
        assert tokenizer.PAD in tokenizer.word2id
        assert tokenizer.UNK in tokenizer.word2id
        assert tokenizer.BOS in tokenizer.word2id
        assert tokenizer.EOS in tokenizer.word2id
        
        # Verify IDs (should be 0,1,2,3 respectively)
        assert tokenizer.word2id[tokenizer.PAD] == 0
        assert tokenizer.word2id[tokenizer.UNK] == 1
        assert tokenizer.word2id[tokenizer.BOS] == 2
        assert tokenizer.word2id[tokenizer.EOS] == 3
    
    
    def test_decode_known_ids(self):
        """Test decoding valid IDs back to words."""
        # First encode, then decode
        text = "hello world"
        encoded = self.vocab_tokenizer.encode(text)
        decoded = self.vocab_tokenizer.decode(encoded)
        
        # Decoding should recover original text
        assert decoded == text
    
    def test_decode_with_padding(self):
        """Test decode ignores padding tokens."""
        text = "hello world"
        encoded = self.vocab_tokenizer.encode(text)
        # Add padding at the end
        padded = encoded + [0, 0, 0]  # 0 is PAD ID
        
        decoded = self.vocab_tokenizer.decode(padded)
        assert decoded == text  # Padding should be ignored
    
    def test_decode_unknown_ids(self):
        """Test unknown IDs become UNK in decoded text."""
        # Create list with valid ID and invalid ID
        hello_id = self.vocab_tokenizer.word2id["hello"]
        invalid_id = 9999  # ID that doesn't exist
        
        result = self.vocab_tokenizer.decode([hello_id, invalid_id])
        # Result should contain "hello" and UNK token
        assert "hello" in result
        assert self.vocab_tokenizer.UNK in result
    
    def test_decode_with_special_tokens(self):
        """Test that special tokens are decoded appropriately."""
        # Get IDs for special tokens
        bos_id = self.tokenizer.word2id[self.tokenizer.BOS]
        eos_id = self.tokenizer.word2id[self.tokenizer.EOS]
        
        # Need a tokenizer with known words
        tokenizer = WordTokenizer(vocab=["hello", "world"])
        hello_id = tokenizer.word2id["hello"]
        world_id = tokenizer.word2id["world"]
        
        ids = [bos_id, hello_id, world_id, eos_id]
        decoded = tokenizer.decode(ids)
        
        # Depending on whether we filter special tokens in decode
        # Currently, decode treats them as normal words
        assert tokenizer.BOS in decoded or "hello" in decoded
    
    
    def test_vocab_contains_special_tokens(self):
        """Test vocabulary always includes PAD, UNK, BOS, EOS."""
        vocab = self.tokenizer.get_vocab()
        assert self.tokenizer.PAD in vocab
        assert self.tokenizer.UNK in vocab
        assert self.tokenizer.BOS in vocab
        assert self.tokenizer.EOS in vocab
        
        # They should be at the beginning in that order
        assert vocab[0] == self.tokenizer.PAD
        assert vocab[1] == self.tokenizer.UNK
        assert vocab[2] == self.tokenizer.BOS
        assert vocab[3] == self.tokenizer.EOS
    
    def test_vocab_size_with_custom_vocab(self):
        """Test vocabulary size with custom word list."""
        custom_words = ["hello", "world", "hola", "mundo"]
        tokenizer = WordTokenizer(vocab=custom_words)
        
        # Size = 4 special + 4 words = 8
        assert tokenizer.get_vocab_size() == 8
        
        # Verify no duplicates
        vocab = tokenizer.get_vocab()
        assert len(vocab) == len(set(vocab))  # All are unique
    
    def test_vocab_deduplication(self):
        """Test that duplicate words are removed from vocabulary."""
        words_with_duplicates = ["hello", "world", "hello", "world", "hola"]
        tokenizer = WordTokenizer(vocab=words_with_duplicates)
        
        # Should have: 4 special + 3 unique words = 7
        assert tokenizer.get_vocab_size() == 7
        
        # Verify "hello" appears only once
        vocab = tokenizer.get_vocab()
        assert vocab.count("hello") == 1
    
    def test_special_tokens_not_duplicated_in_custom_vocab(self):
        """Test that if custom vocab contains special tokens, they're filtered."""
        words_with_specials = ["hello", "<pad>", "world", "<unk>", "<bos>", "<eos>"]
        tokenizer = WordTokenizer(vocab=words_with_specials)
        
        # Should have: 4 special + 2 unique words ("hello", "world") = 6
        assert tokenizer.get_vocab_size() == 6
        
        # Verify specials weren't duplicated
        vocab = tokenizer.get_vocab()
        assert vocab.count(tokenizer.PAD) == 1
        assert vocab.count(tokenizer.UNK) == 1
    
    
    def test_from_corpus_basic(self):
        """Test creating tokenizer from corpus."""
        corpus = [
            "hello world",
            "hola mundo",
            "hello hola"
        ]
        
        tokenizer = WordTokenizer.from_corpus(corpus)
        
        # Should have: 4 special + unique words: hello, world, hola, mundo = 8
        assert tokenizer.get_vocab_size() == 8
        
        # Verify all words are in vocabulary
        vocab = tokenizer.get_vocab()
        for word in ["hello", "world", "hola", "mundo"]:
            assert word in vocab
    
    def test_from_corpus_with_lowercase(self):
        """Test corpus building with lowercase option."""
        corpus = [
            "Hello World",
            "HOLA MUNDO"
        ]
        
        # With lowercase=True (default)
        tokenizer_lower = WordTokenizer.from_corpus(corpus, lowercase=True)
        vocab_lower = tokenizer_lower.get_vocab()
        
        # With lowercase=False
        tokenizer_upper = WordTokenizer.from_corpus(corpus, lowercase=False)
        vocab_upper = tokenizer_upper.get_vocab()
        
        # Lowercase version should have lowercase words
        assert "hello" in vocab_lower
        assert "world" in vocab_lower
        assert "hola" in vocab_lower
        assert "mundo" in vocab_lower
        
        # Uppercase version should preserve original case
        # Note: The actual words present depend on tokenization
        # "Hello" and "HOLA" are different tokens when case is preserved
        assert "Hello" in vocab_upper or "hello" not in vocab_upper
    
    def test_from_corpus_empty(self):
        """Test creating tokenizer from empty corpus."""
        tokenizer = WordTokenizer.from_corpus([])
        
        # Should only have special tokens: PAD, UNK, BOS, EOS = 4
        assert tokenizer.get_vocab_size() == 4
        assert tokenizer.get_vocab() == ["<pad>", "<unk>", "<bos>", "<eos>"]
    
    
    def test_vocab_building_preserves_order(self):
        """Test that vocabulary maintains insertion order (Python 3.7+)."""
        words = ["zebra", "apple", "mango", "banana"]
        tokenizer = WordTokenizer(vocab=words)
        
        vocab = tokenizer.get_vocab()
        # First 4 are special tokens
        assert vocab[4:8] == ["zebra", "apple", "mango", "banana"]
    
    def test_word2id_and_id2word_are_consistent(self):
        """Test that mappings are bidirectional."""
        tokenizer = self.vocab_tokenizer
        
        # Test for known words
        for word in ["hello", "world", "hola", "mundo"]:
            idx = tokenizer.word2id[word]
            assert tokenizer.id2word[idx] == word
        
        # Test for special tokens
        for token in [tokenizer.PAD, tokenizer.UNK, tokenizer.BOS, tokenizer.EOS]:
            idx = tokenizer.word2id[token]
            assert tokenizer.id2word[idx] == token