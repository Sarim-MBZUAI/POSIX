import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
import random
import string
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

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
        
        # randomly sample tokens to introduce error into
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
        # Introducing juxtaposition by replacing letters with the adjacent letters according to keyboard
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
    
class SynonymPerturber:
    def __init__(self, n_error):
        self.n_error = n_error
        
        # DT: determiner; NNP: singular proper noun; PRP: personal pronoun; PRP$: possessive pronoun; POS: genitive marker; CD: numeral/cardinal
        self.invalid_pos_tags = ['DT', 'NNP', 'PRP', 'PRP$', 'POS', 'CD']
        self.stopwords = set(stopwords.words('english'))
        self.min_word_length = 3
        self.max_word_length = 15

    def perturb_sentence(self, sentence):
        tokens = word_tokenize(sentence)

        valid_tokens=[]
        valid_indices=[]
        for i, token in enumerate(tokens):
            if self._is_valid_word(token):
                valid_tokens.append(token)
                valid_indices.append(i)
        
        # Select n_error unique indices randomly
        indices_to_replace = random.sample(valid_indices, min(self.n_error, len(valid_tokens)))
        replaced_count = 0
        
        # Replace tokens at selected indices
        for index in indices_to_replace:
            tokens[index] = self._replace_word(tokens[index])
            replaced_count += 1
            
            # If all n replacements have been made, break the loop
            if replaced_count >= self.n_error:
                break
        
        return ' '.join(tokens)
    
    def _get_synonym(self, word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return synonyms

    def _replace_word(self, word):
        synonyms = self._get_synonym(word)
        if synonyms:
            return random.choice(synonyms)
        else:
            return word

    def _is_valid_word(self, word):
        # Check if word is in stopwords
        if word.lower() in self.stopwords:
            return False

        # Check word length: short words might not provide meaningful synonyms, and very long words might be domain-specific or technical terms.
        if len(word) < self.min_word_length or len(word) > self.max_word_length:
            return False

        # words with non-alphabetic characters 
        if not word.isalpha():
            return False
        
        # Potential proper noun
        if word[0].isupper():
            return False

        # Check if word is punctuation or number
        if word in string.punctuation or word.isdigit():
            return False

        # Check if word's part of speech is in the list of invalid POS tags
        pos = pos_tag([word])[0][1]
        if pos in self.invalid_pos_tags:
            return False
        
        return True