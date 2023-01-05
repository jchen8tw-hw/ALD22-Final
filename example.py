from implicit.datasets.lastfm import get_lastfm
from implicit.nearest_neighbours import bm25_weight
from implicit.gpu.als import AlternatingLeastSquares

artists, users, artist_user_plays = get_lastfm()



# weight the matrix, both to reduce impact of users that have played the same artist thousands of times
# and to reduce the weight given to popular items
artist_user_plays = bm25_weight(artist_user_plays, K1=100, B=0.8)

# get the transpose since the most of the functions in implicit expect (user, item) sparse matrices instead of (item, user)
user_plays = artist_user_plays.T.tocsr()

model = AlternatingLeastSquares(factors=64, regularization=0.05)
model.fit(user_plays)
userid = 12345
# print(user_plays[userid])
ids, scores = model.recommend(userid, user_plays[userid], N=10, filter_already_liked_items=False)
