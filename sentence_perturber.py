import nltk
from nltk.tokenize import word_tokenize
import random
nltk.download('punkt')

class NullPerturber:
    def __init__(self):
        pass

    def perturb_sentence(self, sentence):
        return sentence

class GrammaticalPerturber:
    def __init__(self, n_error, error_type):
        self.n_error = n_error
        self.error_type = error_type # one among add, del, sub, jux, mix

    def perturb_sentence(self, sentence):
        tokens = word_tokenize(sentence)
        num_errors = self.n_error
        
        #randomly sample tokens to introduce error into
        #random.seed(42)
        N_tokens=len(tokens)
        error_indices = random.sample(range(N_tokens), min(num_errors, N_tokens))

        perturbed_tokens = []
        for i, token in enumerate(tokens):
            if i in error_indices:
                perturbed_tokens.append(self._introduce_grammar_error(token))
            else:
                perturbed_tokens.append(token)

        return ' '.join(perturbed_tokens)
    
    def _introduce_grammar_error(self, token):
    
        # Introducing the selected error
        if self.error_type == 'add':
            return self._introduce_addition(token)
        elif self.error_type == 'del':
            return self._introduce_deletion(token)
        elif self.error_type == 'sub':
            return self._introduce_substitution(token)
        elif self.error_type == 'jux':
            return self._introduce_juxtaposition(token)
        elif self.error_type == 'mix': 
            return self._introduce_mix_error(token)
        else:
            return token
    
    def _introduce_mix_error(self, token):
        error_types = ['add', 'del', 'sub', 'jux']

        # Selecting a random error type
        error = random.choice(error_types)

        if error == 'add':
            return self._introduce_addition(token)
        elif error == 'del':
            return self._introduce_deletion(token)
        elif error == 'sub':
            return self._introduce_substitution(token)
        elif error == 'jux':
            return self._introduce_juxtaposition(token)
        else:
            return token
    
    def _introduce_addition(self, token):
        # Introducing grammatical error by adding a random character
        if len(token) > 1:
            index = random.randrange(1, len(token)) 
            return token[:index] + random.choice('abcdefghijklmnopqrstuvwxyz') + token[index:]
        else:
            return token
    
    def _introduce_deletion(self, token):
        # Introducing grammatical error by deleting a random letter
        if len(token) > 2:
            index = random.randrange(1, len(token)-1) 
            return token[:index] + token[index+1:]
        else:
            return token
    
    def _introduce_substitution(self, token):
        # Introducing substitution by randomly swapping adjacent letters
        if len(token) > 2:
            index = random.randrange(len(token)-1) 
            return token[:index] + token[index + 1] + token[index] + token[index+2:]
        else:
            return token
    
    def _introduce_juxtaposition(self, token):
        # Introducing juxtraposition by replacing letters with the adjacent letters according to keyboard
        keyboard_layout = {
            'q': 'wsa',
            'w': 'qasde',
            'e': 'wsdfr',
            'r': 'edfgt',
            't': 'rfghy',
            'y': 'tghju',
            'u': 'yhjki',
            'i': 'ujklo',
            'o': 'iklp',
            'p': 'ol',
            'a': 'qwsxz',
            's': 'qwedcxza',
            'd': 'wersfvxc',
            'f': 'ertdgcvb',
            'g': 'rtyfhvbn',
            'h': 'tyugjbcnm',
            'j': 'yuihknm',
            'k': 'uioljmn',
            'l': 'iopkm',
            'z': 'asx',
            'x': 'zsdc',
            'c': 'xdfv',
            'v': 'cfgb',
            'b': 'vghn',
            'n': 'bhjm',
            'm': 'njk',
        }

        # Select a random index within the word
        index = random.randint(0, len(token) - 1)

        # Introduce mistyping error for the selected character
        if token[index].lower() in keyboard_layout:
            return token[:index] + random.choice(keyboard_layout[token[index].lower()]) + token[index+1:]
        else:
            return token
    
