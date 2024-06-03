import string
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud


f = open("books/carrol/alice_in_wonderland.txt", "r", encoding="utf-8")
text = f.read()

text = text.lower()
spec_chars = string.punctuation
text = "".join([ch for ch in text if ch not in spec_chars])
text = re.sub("\n", " ", text)


text_tokens = word_tokenize(text)
text = nltk.Text(text_tokens)

english_stopwords = stopwords.words("english")
english_stopwords.extend(["one", "im", "said"])

# exclude the stopwords from tokens
text_tokens = [token.strip() for token in text_tokens if token not in english_stopwords]

text = nltk.Text(text_tokens)

# FreqDist calculates the frequency of each element in the dataset.
# fdist_sw = FreqDist(text)

# print the most common words
# print(fdist_sw.most_common(5))

text_raw = " ".join(text)

# create the wordcloud
wordcloud = WordCloud(
    width=1600, height=800, background_color="black", colormap="gist_rainbow_r"
).generate(text_raw)

# create matplotlib object
plt.figure(figsize=(20, 10), facecolor="r")

# add the wordcloud
plt.imshow(wordcloud)

# turn off axis
plt.axis("off")

# turn off border
plt.tight_layout(pad=0)

# display the tag cloud
plt.show()
