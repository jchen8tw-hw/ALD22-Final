import pandas as pd
from typing import *
from  functools import total_ordering
import queue as q
from average_precision import mapk


@total_ordering
class Node:
    def __init__(self, course_id : str):
        self.course_id : str = course_id
        self.children : Dict[str, Node] = {}
        self.total_path : int = 1
    def __eq__(self, other):
        return self.total_path == other.total_path
    # we want the larger one to be the first one
    # therefore, larger is smaller
    def __lt__(self, other):
        return self.total_path > other.total_path

class BayesianTree:
    def __init__(self):
        self.start_courses : Dict[str, Node] = {}
    def predict(self,bought_courses : List[str]) -> List[str]:
        most_probs : q.PriorityQueue = q.PriorityQueue()
        current_node : Node = self.start_courses[bought_courses[0]]
        for i in range(1,len(bought_courses)):
            if bought_courses[i] not in current_node.children:
                raise ValueError("The course is not in the tree")
            current_node = current_node.children[bought_courses[i]]
        for course in current_node.children.values():
            most_probs.put(course)
        predict_courses : List[str] = []
        while not most_probs.empty():
            most_prob_course : Node = most_probs.get()
            predict_courses.append(most_prob_course.course_id)
            for course in most_prob_course.children.values():
                most_probs.put(course)
        return predict_courses

def build_tree(train: pd.DataFrame) -> BayesianTree:
    tree : BayesianTree = BayesianTree()
    for _, course_ids in train.iterrows():
        bought_courses : List[str] = course_ids["course_id"].split(' ')
        if bought_courses[0] not in tree.start_courses:
            tree.start_courses[bought_courses[0]] = Node(bought_courses[0])
        else:
            tree.start_courses[bought_courses[0]].total_path += 1
        current_node : Node = tree.start_courses[bought_courses[0]]
        for i in range(1,len(bought_courses)):
            if bought_courses[i] not in current_node.children:
                current_node.children[bought_courses[i]] = Node(bought_courses[i])
            else:
                current_node.children[bought_courses[i]].total_path += 1
            current_node = current_node.children[bought_courses[i]]
    return tree



if __name__ == '__main__':
    train : pd.DataFrame = pd.read_csv('./data/train.csv',index_col="user_id")
    valid: pd.DataFrame = pd.read_csv('./data/val_seen.csv',index_col="user_id")
    test: pd.DataFrame = pd.read_csv('./data/test_seen.csv',index_col="user_id")
    prob_tree : BayesianTree = build_tree(train=train)

    # validation
    predict_list : List[List[str]] = []
    valid_list : List[List[str]] = []
    for user_id, course_ids in valid.iterrows():
        bought_courses : List[str] = train.loc[user_id]["course_id"].split(' ')
        predict_courses : List[str] = prob_tree.predict(bought_courses)
        predict_list.append(predict_courses)
        valid_list.append(course_ids["course_id"].split(' '))
    print(mapk(valid_list,predict_list,k=728))

    # predict
    for user_id, course_ids in test.iterrows():
        bought_courses : List[str] = train.loc[user_id]["course_id"].split(' ')
        predict_courses : List[str] = prob_tree.predict(bought_courses)
        test.loc[user_id]["course_id"] = ' '.join(predict_courses)
    test.to_csv("./data/predict.csv",index_label="user_id")

