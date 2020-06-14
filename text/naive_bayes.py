import math
from typing import List, Dict, Tuple

class NaiveBayes:
    def __init__(self, n_categories: int, labeled_documents: List[Tuple[List[str], int]]):
        self.n_categories = n_categories

        words = [word for doc in labeled_documents for word in doc[0]]
        n = len(labeled_documents)

        # 先验概率
        self.p_category = [len([doc for doc in labeled_documents if doc[1] == i]) / n for i in range(n_categories)]

        count_word_in_category = { word: [0] * n_categories for word in words }
        count_all_words_in_category = [0] * n_categories
        for doc_words, category in labeled_documents:
            for word in doc_words:
                count_word_in_category[word][category] += 1
                count_all_words_in_category[category] += 1

        # 条件概率
        self.p_word_in_category: Dict[str, List[float]] = {
            word: [
                (count_word_in_category[word][i] + 1) / (count_all_words_in_category[i] + len(words))
                for i in range(n_categories)
            ]
            for word in words
        }

        # 如果某个单词没有出现在训练集中，则为 1 / (count_all_words_in_category[i] + len(words))
        self.p_word_in_category_if_not_exist = [1 / (count_all_words_in_category[i] + len(words)) for i in range(n_categories)]
    
    # 预测一个文档的类别
    def predict(self, document: List[str]) -> int:
        max_p_category = None
        max_p = None

        for i in range(self.n_categories):
            p = math.log(self.p_category[i])
            for word in document:
                if word in self.p_word_in_category:
                    p += math.log(self.p_word_in_category[word][i])
                else:
                    p += math.log(self.p_word_in_category_if_not_exist[i])

            if max_p is None or p > max_p:
                max_p_category = i
                max_p = p
        
        return max_p_category

            