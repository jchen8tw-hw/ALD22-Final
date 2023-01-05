import pandas as pd
from implicit.nearest_neighbours import bm25_weight
from implicit.gpu.als import AlternatingLeastSquares
import numpy as np
from scipy.sparse import csr_matrix
from average_precision import mapk

courses = pd.read_csv("./data/courses.csv")["course_id"].to_numpy()
course2index = {}
for i in range(len(courses)):
    course2index[courses[i]] = i
users = pd.read_csv("./data/users.csv")["user_id"].to_numpy()
user2index = {}
for i in range(len(users)):
    user2index[users[i]] = i
train = pd.read_csv("./data/train.csv")
val_seen = pd.read_csv("./data/val_seen.csv",index_col="user_id")
test = pd.read_csv("./data/test_seen.csv",index_col="user_id")
user_buy_course = np.zeros((len(users), len(courses)))
for i in range(len(train)):
    courseids = train.iloc[i]["course_id"].split(" ")
    for courseid in courseids:
        user_buy_course[user2index[train.iloc[i]["user_id"]]][course2index[courseid]] = 1
user_buy_course = csr_matrix(user_buy_course)

#optimize find best K1 and B
best_mapk = 0
best_K1 = 0
best_fact   = 0
actual_ids = []
best_pred_courses_id = None
for uuid in val_seen.index:
        actual = val_seen.loc[uuid]["course_id"].split(" ")
        actual_ids.append(actual)
for K1 in range(10,100,5):
    for fact in np.arange(32,256,32):
        user_buy_course_bm25 = bm25_weight(user_buy_course, K1=K1, B=0.1).tocsr()
        model = AlternatingLeastSquares(factors=fact, regularization=0.06,iterations=30)
        model.fit(user_buy_course_bm25)
        pred_ids = []
        userids = []
        for uuid in val_seen.index:
            userids.append(user2index[uuid])
        userids = np.array(userids)
        ids, scores = model.recommend(userids, user_buy_course_bm25[userids], N=728, filter_already_liked_items=True)
        pred_courses_id = []
        for user_row in ids:
            pred_courses_id.append(list(map(lambda id: courses[id],user_row)))
        mapk_score = mapk(actual_ids, pred_courses_id, k=728)
        if mapk_score > best_mapk:
            best_mapk = mapk_score
            best_K1 = K1
            best_fact = fact
            best_pred_courses_id = pred_courses_id
            print("K1:",K1,"fact:",fact,"mapk:",mapk_score)
print("best K1:",best_K1,"best fact:",best_fact,"best mapk:",best_mapk)


# predict
user_buy_course_bm25 = bm25_weight(user_buy_course, K1=best_K1, B=0.1).tocsr()
model = AlternatingLeastSquares(factors=best_fact, regularization=0.06,iterations=30)
model.fit(user_buy_course_bm25)
pred_ids = []
userids = []
for uuid in test.index:
    userids.append(user2index[uuid])
userids = np.array(userids)
ids, scores = model.recommend(userids, user_buy_course_bm25[userids], N=728, filter_already_liked_items=True)
pred_courses_id = []
for user_row in ids:
    pred_courses_id.append(list(map(lambda id: courses[id],user_row)))

for i in range(len(test)):
    test.iloc[i]["course_id"] = " ".join(pred_courses_id[i])
# save file
test.to_csv("./data/predict.csv",index_label="user_id")

# user_buy_course_bm25 = bm25_weight(user_buy_course, K1=100, B=0.8).tocsr()
# model = AlternatingLeastSquares(factors=64, regularization=0.05)
# model.fit(user_buy_course_bm25)

# # validate
# actual_ids = []
# pred_ids = []
# for uuid in val_seen.index:
#     userid = user2index[uuid]
#     ids, scores = model.recommend(userid, user_buy_course_bm25[userid], N=728, filter_already_liked_items=True)
#     pred_courses_id = []
#     for id in ids:
#         pred_courses_id.append(courses[id])
#     pred_ids.append(pred_courses_id)
#     actual = val_seen.loc[uuid]["course_id"].split(" ")
#     actual_ids.append(actual)
# print(mapk(actual_ids, pred_ids, k=728))

# predict
# for uuid in test.index:
#     userid = user2index[uuid]
#     ids, scores = model.recommend(userid, user_buy_course_bm25[userid], N=728, filter_already_liked_items=True)
#     # print(ids,scores)
#     courses_id = []
#     for id in ids:
#         courses_id.append(courses[id])
#     test.loc[uuid]["course_id"] = " ".join(courses_id)

# test.to_csv("./data/predict.csv",index_label="user_id")


