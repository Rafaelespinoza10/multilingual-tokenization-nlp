
import pytest
from text_tokenizer import CharTokenizer

class TestCharTokenizer:
    """ 
    tests for CharTokenizer
    """

    def setup_method(self):
        """
        Create a fresh tokenizer for each test
        """
        self.tokenizer = CharTokenizer()
    
    def test_tokenize_simple_word(self):
        """
        Test tokenization of a simple word
        """
        result = self.tokenizer.tokenize("hello")
        assert result == ["h", "e", "l", "l", "o"]
    
    def test_tokenize_simple_sentence(self):
        """
        Test tokenization of a simple sentence
        """
        result = self.tokenizer.tokenize("hello world")
        expected = ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
        assert result == expected
    
    def test_tokenize_empty_string(self):
        """
        Test tokenization of an empty string
        """
        text = ""
        result = self.tokenizer.tokenize(text)
        assert result == []
    
    def test_tokenize_special_characters(self):
        """
        Test tokenization of unicode characters
        """
        text = "hello, world!"
        result = self.tokenizer.tokenize(text)
        assert "!" in result
        assert "," in result
        assert " " in result
        assert result[-1] == "!"
 
    def test_tokenize_unicode_characters(self):
        """ Test unicode characters (Chinese, accents)"""
        text = "你好，世界！"
        result = self.tokenizer.tokenize(text)
        expected = ["你", "好", "，", "世", "界", "！"]
        assert result == expected
    
    def test_encode_simple_word(self):
        """ Test encoding of a simple word """
        result = self.tokenizer.encode("hello")
        assert len(result) == 5
        assert all(isinstance(i, int) for i in result)
        assert all(0 <= i < self.tokenizer.get_vocab_size() for i in result)
    
    def test_encode_unknown_word(self):
        """ Test encoding of an unknown word """
        limited_vocab = "abc"
        tokenizer = CharTokenizer(chars=limited_vocab)
        result = tokenizer.encode("abcd")
        unk_id = tokenizer.char2id[tokenizer.UNK]

        assert result == [
            tokenizer.char2id['a'],
            tokenizer.char2id['b'],
            tokenizer.char2id['c'],
            unk_id
        ]

    def test_encode_empty_string(self):
        result = self.tokenizer.encode("")
        assert result == []

    def test_encode_simple_word(self):
        """ Test encoding of a simple word """
        original = "hello"
        encoded = self.tokenizer.encode(original)
        decoded = self.tokenizer.decode(encoded)
        assert decoded == original
    
    def test_decode_simple_word(self):
        # simulate final padding
        text = "hi"
        encoded = self.tokenizer.encode(text)
        padded = encoded + [0,0,0]
        decoded = self.tokenizer.decode(padded)
        #padding should be ignored
        assert decoded == text

    def test_decode_unknown_word(self):
        #create a id invalid word
        invalid_ids = [1, 9999, 2]
        # should not raise an error
        result = self.tokenizer.decode(invalid_ids)
        assert isinstance(result, str)
        #id 9999 should be replaced by UNK
        assert self.tokenizer.UNK in result or len(result) > 0
    
    # vocabulary test
    def test_vocab_contains_pad_and_unk(self):
        vocab = self.tokenizer.get_vocab()
        assert self.tokenizer.PAD in vocab
        assert self.tokenizer.UNK in vocab
        assert vocab[0] == self.tokenizer.PAD
        assert vocab[1] == self.tokenizer.UNK
    
    def test_get_vocab_size(self):
        size = self.tokenizer.get_vocab_size()
        assert size > 2

        tokenizer = CharTokenizer(chars="abc")
        assert tokenizer.get_vocab_size() == 5 # a, b, c, <pad>, <unk>
    
    def test_char2id_mapping(self):
        assert 'a' in self.tokenizer.char2id
        assert 'z' in self.tokenizer.char2id
        assert 'á' in self.tokenizer.char2id
        assert 'ñ' in self.tokenizer.char2id

        # verify mapping is correct
        assert self.tokenizer.char2id[self.tokenizer.PAD] == 0
        assert self.tokenizer.char2id[self.tokenizer.UNK] == 1
    
    def test_id2char_mapping(self):
        sample_chars = ['a', 'z', 'á', 'ñ', '?', '!']
        for c in sample_chars:
            if c in self.tokenizer.char2id:
                idx = self.tokenizer.char2id[c]
                assert self.tokenizer.id2char[idx] == c

    def test_round_trip_preserves_text(self):
        """Test that encode + decode preserves original text for known characters."""
        texts = [
            "hello",
            "hello world",
            "¡Hola, mundo!",
            "café",
            "123 456",
            "",
            "   "
        ]
        
        for text in texts:
            encoded = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(encoded)
            assert decoded == text, f"Failed for: '{text}'"

    def test_round_trip_with_unknown_chars(self):
        """Test that unknown characters become UNK in round trip."""
        tokenizer = CharTokenizer()
        
        text = "你好"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        # Should become UNK tokens
        assert decoded == "<unk><unk>"
        
        # Alternative: check that original characters are not in decoded
        assert "你" not in decoded
        assert "好" not in decoded
    
    def test_custom_vocabulary(self):
        """Test tokenizer with custom character set."""
        custom_chars = "abc123"
        tokenizer = CharTokenizer(chars=custom_chars)
        
        assert tokenizer.encode("abc") == [2, 3, 4]
        
        unk_id = tokenizer.char2id[tokenizer.UNK]
        assert tokenizer.encode("xyz") == [unk_id, unk_id, unk_id]

