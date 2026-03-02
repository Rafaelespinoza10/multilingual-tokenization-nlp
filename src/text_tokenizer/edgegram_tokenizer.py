from typing import List, Optional, Dict, Union
from text_tokenizer.base import BaseTokenizer

class EdgegramTokenizer(BaseTokenizer):
    """
    Tokenizer based on the Edgegram model.
    eg. "hello" with n=3 generates:
     - Prefix : ["hel"]
     - Suffix : ["llo"]
    Depends on the settings
    """

    def __init__(self, n: int = 3, use_prefix: bool = True, use_suffix: bool = True, language: Optional[str] = None):
        """
        Args: 
            n: length of the edgegram
            use_prefix: whether to use the prefix
            use_suffix: whether to use the suffix
            language: language of the text (passed to base class)
        """
        # Call parent constructor if it accepts language
        super().__init__()  # BaseTokenizer doesn't take parameters
        
        self.n = n
        self.use_prefix = use_prefix
        self.use_suffix = use_suffix
        self.language = language  # Store language even if base doesn't use it
        
        if not use_prefix and not use_suffix:
            raise ValueError("At least one of use_prefix or use_suffix must be True")
        
        # Vocabulary management
        self.PAD = "<pad>"
        self.UNK = "<unk>"
        self.vocab = [self.PAD, self.UNK]  # Start with special tokens
        self.token2id = {self.PAD: 0, self.UNK: 1}
        self.id2token = {0: self.PAD, 1: self.UNK}
    
    def tokenize(self, text: str) -> List[str]:
        """
        Convert text to edge-gram tokens.
        
        Args:
            text: Input string
            
        Returns:
            List of edge-gram tokens
        """
        if not text or not text.strip():
            return []
        
        words = text.split()
        edgegrams = []
        
        for word in words:
            if len(word) < self.n:
                # Words shorter than n are kept as-is
                edgegrams.append(word)
                continue
            
            if self.use_prefix:
                prefix = word[:self.n]
                edgegrams.append(prefix)
            
            if self.use_suffix:
                suffix = word[-self.n:]
                edgegrams.append(suffix)
        
        return edgegrams
    
    def _add_tokens_to_vocab(self, tokens: List[str]) -> None:
        """
        Add new tokens to vocabulary if they don't exist.
        
        Args:
            tokens: List of tokens to potentially add
        """
        for token in tokens:
            if token not in self.token2id:
                idx = len(self.vocab)
                self.vocab.append(token)
                self.token2id[token] = idx
                self.id2token[idx] = token
    
    def encode(self, text: str, add_to_vocab: bool = True) -> List[int]:
        """
        Convert text to list of token IDs.
        
        Args:
            text: Input string
            add_to_vocab: If True, add new tokens to vocabulary. 
                        If False, unknown tokens become UNK.
        
        Returns:
            List of integer IDs corresponding to tokens
        """
        tokens = self.tokenize(text)
        
        if add_to_vocab:
            self._add_tokens_to_vocab(tokens)
        
        return [self.token2id.get(token, self.token2id[self.UNK]) for token in tokens]
        
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            String representation (tokens joined by spaces)
        """
        tokens = []
        for i in ids:
            if i in self.id2token:
                token = self.id2token[i]
                if token != self.PAD:  # Skip padding tokens
                    tokens.append(token)
            else:
                tokens.append(self.UNK)
        
        return " ".join(tokens)
    
    def get_vocab_size(self) -> int:
        """
        Return the size of the vocabulary.
        
        Returns:
            Number of tokens in vocabulary
        """
        return len(self.vocab)
    
    def get_vocab(self) -> Union[List[str], Dict]:
        """
        Return the full vocabulary.
        
        Returns:
            List of all tokens in vocabulary
        """
        return self.vocab.copy()
    
    # Additional helper methods (not required by base class but useful)
    
    def get_token2id(self) -> Dict[str, int]:
        """Return token to ID mapping."""
        return self.token2id.copy()
    
    def get_id2token(self) -> Dict[int, str]:
        """Return ID to token mapping."""
        return self.id2token.copy()
    
    def reset_vocabulary(self):
        """Reset vocabulary to only PAD and UNK."""
        self.vocab = [self.PAD, self.UNK]
        self.token2id = {self.PAD: 0, self.UNK: 1}
        self.id2token = {0: self.PAD, 1: self.UNK}