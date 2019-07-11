# kaggle datasets download -d rush4ratio/video-game-sales-with-ratings --unzip

import numpy as np
import pandas as pd

df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

df = df.dropna(subset = ['Genre', 'Publisher']).reset_index()
df['Criteria'] = df['Genre'].str.cat(df[['Platform', 'Publisher']], sep = ' ')

# count genre
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(
    tokenizer = lambda i: i.split(' '),    # => cari split karakter yg unik di luar feature
    analyzer = 'word'
)
matrix_genre = model.fit_transform(df['Criteria'])
tipe_genre = model.get_feature_names()
jumlah_genre = len(tipe_genre)
event_genre = matrix_genre.toarray()

# print(tipe_genre)
# print(jumlah_genre)

# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(matrix_genre)
# print(score)

# test model
saya_suka = 'Suikoden'
print(df[df['Name'] == 'Suikoden'])

# take the index from saya_suka
index_suka = df[df['Name'] == saya_suka].index.values[0]

# list all games + cosine similarity score
all_games = list(enumerate(score[index_suka]))

# show 5 first datas, sorted by index
game_sama = sorted(
    all_games,
    key = lambda i: i[1],
    reverse = True
)

# for i in game_sama[:5]:
#     print(
#         df.iloc[i[0]]['Name'],
#         df.iloc[i[0]]['Platform'],
#         df.iloc[i[0]]['Genre'],
#         df.iloc[i[0]]['Publisher'],
#         )

# list all games filter by cosine similarity score > 80%
game_80up = []
for i in game_sama:
    if i[1] > 0.8:
        game_80up.append(i)

# show 5 datas randomly, where cosine similarity score > 50%
import random
rekomendasi = random.choices(game_80up, k = 5)

for i in rekomendasi:
    print(
        df.iloc[i[0]]['Name'],
        df.iloc[i[0]]['Platform'],
        df.iloc[i[0]]['Genre'],
        df.iloc[i[0]]['Publisher']
    )
