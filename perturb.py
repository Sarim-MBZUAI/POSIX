import nltk
from nltk.tokenize import word_tokenize
import random
import csv
import argparse
nltk.download('punkt')

def parse_args():
    parser=argparse.ArgumentParser(description="Perturb a given set of prompts")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="input.csv",
        help="The input csv file"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="output.csv",
        help="The output csv file"
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="Sentences",
        help="The name of the column which contains the input prompts"
    )
    parser.add_argument(
        "--error_percentage",
        type=float,
        default=20,
        help="The percentage of sentences to perturb"
    )
    args=parser.parse_args()
    return args

class SentencePerturber:
    def __init__(self, input_file_path, output_file_path, column_name, n_error):
        self.input_file = input_file_path
        self.output_file = output_file_path
        self.column_name = column_name
        self.n_error = n_error

    def perturb_sentences(self):
        
        with open(self.input_file, 'r', encoding='utf-8') as csvfile:
            reader = list(csv.reader(csvfile))
          
            with open(self.output_file, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    col_index = reader[0].index(self.column_name)
                    #print(col_index)
                    writer.writerow(reader[0])
                    for row in reader[1:]:
                        #print(row)
                        perturbed_sentence = self._perturb_sentence(row[col_index])
                        row[col_index] = perturbed_sentence
                        writer.writerow(row)

    def _perturb_sentence(self, sentence):
        tokens = word_tokenize(sentence)
        num_errors = self.n_error
        
        #randomly sample tokens to introduce error into
        #random.seed(42)
        error_indices = random.sample(range(len(tokens)), num_errors)

        perturbed_tokens = []
        for i, token in enumerate(tokens):
            if i in error_indices:
                perturbed_tokens.append(self._introduce_grammar_error(token))
            else:
                perturbed_tokens.append(token)

        return ' '.join(perturbed_tokens)

    def _introduce_grammar_error(self, token):
        # List of possible error types
        error_types = ['addition', 'deletion', 'substituiton', 'juxtraposition']

        # Selecting a random error type
        error_type = random.choice(error_types)
    
        # Introducing the selected error
        if error_type == 'addition':
            return self._introduce_addition(token)
        elif error_type == 'deletion':
            return self._introduce_deletion(token)
        elif error_type == 'substituiton':
            return self._introduce_substitution(token)
        elif error_type == 'juxtraposition':
            return self._introduce_juxtraposition(token)
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
    
    def _introduce_juxtraposition(self, token):
        # Introducing substitution by replacing letters with the adjacent letters according to keyboard
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


if __name__ == "__main__":
    args=parse_args()
    perturber = SentencePerturber(args.input_csv, args.output_csv, args.prompt_column, args.error_percentage)
    perturber.perturb_sentences()
