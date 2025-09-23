import re
import contractions
import unicodedata

class Preprocessing:
    def __init__(self, config):
        self.config = config
    
    def select_feature(self, df):
        if 'toxicity' in df.columns:
            df = df[['comment_text', 'toxicity']]
        return df
    
    def structural_cleaning(self, df):
        df_out = df.copy()
        # Drop na rows
        df_out = df_out.dropna()
        # Remove exact dupliates
        df_out = df_out.drop_duplicates()
        # Fix encoding, ensure str type
        df_out['comment_text'] = df_out['comment_text'].astype(str)
        # Ensure float type for the target column
        if 'toxicity' in df.columns:
            df_out['toxicity'] = df_out['toxicity'].astype(float)
        return df_out

    def replace_mention(self, text):
        words = text.split()
        for i in range(len(words)):
            if i != 0:
                if words[i][0].isupper():
                    words[i] = '[MENTION]'
        return ' '.join(words)

    def noise_removal(self, df):
        df_out = df.copy()
        # remove URLs -> [URL]
        df_out['comment_text'] = df_out['comment_text'].apply(lambda x: re.sub(r'http\S+', '<URL>', x))
        # Replace mentions → [MENTION].
        df_out['comment_text'] = df_out['comment_text'].apply(self.replace_mention)
        # Process hashtags (#awesome → awesome).
        df_out['comment_text'] = df_out['comment_text'].apply(lambda x: x.replace('#', ''))
        # strip html
        df_out['comment_text'] = df_out['comment_text'].apply(lambda x: re.sub(r'<.*?>', '<HTML>', x))
        # Remove invisible/non-printable chars
        df_out['comment_text'] = df_out['comment_text'].apply(lambda x: re.sub(r'[\x00-\x1f\ufeff\u200b]', '', x))
        # Replace number with [NUMBER]
        df_out['comment_text'] = df_out['comment_text'].apply(lambda x: re.sub(r'\d+', '<NUMBER>', x))
        return df_out

    def expand_contraction(self, text):
        words = text.split()
        expanded_words = []
        for word in words:
            expanded_words.append(contractions.fix(word, slang = True))
        return ' '.join(expanded_words)
            

    def contraction_slang(self, df):
        df_out = df.copy()
        df_out['comment_text'] = df_out['comment_text'].apply(self.expand_contraction)
        return df_out
    
    def text_normalization(self, df):
        df_out = df.copy()
        # lowercaseing
        df_out['comment_text'] = df['comment_text'].apply(lambda x: x.lower())
        # normalize whitespace
        df_out['comment_text'] = df['comment_text'].apply(lambda x: ' '.join(x.strip().split()))
        # normalize punctuation
        df_out['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        # normalize repeated characters
        df_out['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r'(\w)\1{2,}', r'\1\1', x))
        # Unicode normalization (curly quotes → straight quotes).
        df_out['comment_text'] = df_out['comment_text'].apply(lambda x: unicodedata.normalize('NFKC', x))
        
        return df_out
    
    def run(self, df):
        df = self.select_feature(df)
        df = self.structural_cleaning(df)
        df = self.noise_removal(df)
        df = self.contraction_slang(df)
        df = self.text_normalization(df)
        return df