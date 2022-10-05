import nltk

nltk.download('gutenberg')

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

nltk.download('universal_tagset')

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

from nltk.corpus import gutenberg
from string import punctuation
import matplotlib.pyplot as plt
import re
from collections import Counter

"""## Loading text file"""

gutenberg.fileids()

hamlet = gutenberg.open('shakespeare-hamlet.txt')
hamlet = hamlet.readlines()
hamlet[:5]

"""## Clean unwanted spaces"""

hamlet = list(filter(None, [item.strip('\n') for item in hamlet]))

"""## Text statistics"""

total_char = len(gutenberg.raw('shakespeare-hamlet.txt'))

print(f"total characters in hemlet {total_char}")

total_words = len(gutenberg.words('shakespeare-hamlet.txt'))

print(f"total words in hamlet {total_words}")

total_line = len(gutenberg.sents('shakespeare-hamlet.txt'))

print(f"total number of sentences {total_line}")

longest_line = max(hamlet, key=len)

print(f'longest line in document: {longest_line}')

shortest_line = min(hamlet, key=len)

print(f'shortest line in document: {shortest_line}')

"""## Lingusitics Analysis"""

# cleaning of words
tokens = [item.split() for item in hamlet]
words = [word for sentence in tokens for word in sentence]
words = list(filter(None, [re.sub(r'[^A-Za-z]', '', word) for word in words]))

# tagging of words
pos_tags = nltk.pos_tag(words, tagset='universal', lang='eng')
pos_tags[:10]


# function to find all tag of specific class and show 5 most common
def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                   if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].most_common(10)) for tag in cfd.conditions())


# Find most common Nouns
tagdict_noun = findtags('NOUN', pos_tags)

for tag in sorted(tagdict_noun):
    print(tag, tagdict_noun[tag])

# Find most common Verbs
tagdict_verb = findtags('VERB', pos_tags)

for tag in sorted(tagdict_verb):
    print(tag, tagdict_verb[tag])

# Find most common Adjectives
tagdict_adj = findtags('ADJ', pos_tags)

for tag in sorted(tagdict_adj):
    print(tag, tagdict_adj[tag])

"""## Words per sentence distribution"""

total_tokens_per_line = [len(sentence.split()) for sentence in hamlet]
plt.hist(total_tokens_per_line, color='orange')
plt.show()

"""## Get most common words in data"""

# removing stop words as these used most of the time and these words dont provide any specific details.
words = [word.lower() for word in words if word.lower() not in stopwords]

# getting frequency of all the words and printing top 10 words.
c = Counter(words)
c.most_common(10)
print('most common word:',c.most_common(10))

"""## get list of words based on value count of frequency."""


def get_words(numb):
    words = []
    for key, value in c.items():
        if numb == value:
            words.append(key)

    return words


# checking for exact frequency of 50 words
print('words repeated 50 times',get_words(50))

# creating result.txt file to store the analysis of data

with open('result.txt', 'w') as f:
    f.write('Text analysis of hamlet file\n')
    f.write(f'total characters in hemlet {total_char}\n')
    f.write(f"total words in hamlet {total_words}\n")
    f.write(f"total number of sentences {total_line}\n")
    f.write(f'longest line in document: {longest_line}\n')
    f.write(f'shortest line in document: {shortest_line}\n')
    f.write('\n')
    f.write(f'Noun: {tagdict_noun["NOUN"]}\n')
    f.write(f'Verbs: {tagdict_verb["VERB"]}\n')
    f.write(f'adjective: {tagdict_adj["ADJ"]}\n')
    f.write('\n')
    f.write(f'Most common words used : {c.most_common(10)}')
    f.write(f'Most words with specific occurence : {get_words(50)}')

