import nltk
from nltk.tokenize import word_tokenize
import random
import csv
nltk.download('punkt')

class SentencePerturber:
    def __init__(self, input_file_path, output_file_path, column_name, error_percentage):
        self.input_file = input_file_path
        self.output_file = output_file_path
        self.column_name = column_name
        self.error_percentage = error_percentage

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
        num_errors = round(len(tokens) * self.error_percentage / 100)
        
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
        error_types = ['misspelling', 'letter_swap']

        # Selecting a random error type
        error_type = random.choice(error_types)
    
        # Introducing the selected error
        if error_type == 'misspelling':
            return self._introduce_misspelling(token)
        elif error_type == 'letter_swap':
            return self._introduce_letter_swap(token)
        else:
            return token
    
    def _introduce_misspelling(self, token):
        # Introducing grammatical error by adding a random character
        if len(token) > 1:
            index = random.randrange(1, len(token)) 
            return token[:index] + random.choice('abcdefghijklmnopqrstuvwxyz') + token[index:]
        else:
            return token

    def _introduce_letter_swap(self, token):
        # Introducing letter swap by randomly swapping adjacent letters
        if len(token) > 2:
            index = random.randrange(len(token)-1) 
            return token[:index] + token[index + 1] + token[index] + token[index+2:]
        else:
            return token


if __name__ == "__main__":
    input_csv_path = 'input.csv'
    output_csv_path = 'output.csv'
    column_name = 'Sentences' #Name of the column (in csv) which contains the input prompts to be perturbed
    perturber = SentencePerturber('input.csv', 'output.csv', 'Sentences', 20)  # Error percentage is 10%
    perturber.perturb_sentences()

