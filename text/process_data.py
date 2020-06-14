import argparse
import os
import pickle
import math
import random
from typing import List, Tuple

import utility
from database import Database, Document, Keyword, KeywordOccurrenceInDocument
from naive_bayes import NaiveBayes

# 所有文档，每个文档是一条短信
documents = []
# 在所有文档中出现过的关键词
keywords = {}

# 读取一个有标签或者无标签的数据集
def load_dataset(filename: str, labeled: bool):
    global keywords, documents

    # 当前文档是文件中的第几行，用于显示搜索结果，与算法无关
    line_no = 0
    for line in open(filename, "r"):
        if not line:
            continue

        # 当前文档在整个数据库中的编号
        doc_id = len(documents)
        line_no += 1

        # 输出进度
        if line_no % 100000 == 0:
            print("Loading %s data: %d lines loaded" % ("labeled" if labeled else "non-labeled", line_no))
        
        if labeled:
            label = int(line[0])
            text = line[1:].strip()
        else:
            text = line.strip()
            label = None

        # 切词并移除停用词
        words = utility.cut_words(text)
        if len(words) == 0:
            continue

        doc = Document("%s #%d" % ("Labeled" if labeled else "Non-lebeled", line_no), text, words, label)
        for word in set(words):
            if word not in keywords:
                keywords[word] = Keyword(word)

            # 将该关键词在当前文档中的全部出现添加到倒排索引中
            keywords[word].occurs.append(
                KeywordOccurrenceInDocument(
                    doc_id,
                    list(filter(lambda x: x != None, [i if words[i] == word else None for i in range(len(words))]))
                )
            )

        documents.append(doc)

# 计算 TF-IDF 权重
def calc_tfidf():
    global keywords, documents
    for word in keywords:
        keywords[word].idf = math.log10(1 + len(keywords[word].occurs) / len(documents))
        for occur in keywords[word].occurs:
            occur.tf = len(occur.positions) / len(documents[occur.document_id].words)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labeled_data", help="The data file of labeled SMS texts")
    parser.add_argument("non_labeled_data", help="The data file of non-labeled SMS texts")
    parser.add_argument("database_file", help="The path of file to store the database")
    args = parser.parse_args()
    db_file = open(args.database_file, "wb")

    # 载入已分类数据集
    load_dataset(args.labeled_data, True)
    
    # 将已分类的数据集划分为训练集（90%）和测试集（10%），以测试分类精度
    labeled_documents = list(documents)
    random.shuffle(labeled_documents)
    labeled_count = len(labeled_documents)
    labeled_train_count = int(math.ceil(labeled_count * 0.9))
    labeled_train = labeled_documents[:labeled_train_count]
    labeled_test = labeled_documents[labeled_train_count:]

    # 载入未分类数据集
    load_dataset(args.non_labeled_data, False)
    calc_tfidf()

    # 训练分类器
    classifier = NaiveBayes(2, [(document.words, document.label) for document in labeled_train if document.label is not None])
    # 测试分类器精度
    confusion_matrix = [[0, 0], [0, 0]]
    for document in labeled_test:
        predicted_label = classifier.predict(document.words)
        confusion_matrix[document.label][predicted_label] += 1
    print(
        "Confusion Matrix:\tReal 0\t\tReal 1\n\tPredicted 0\t%d\t\t%d\n\tPredicted 1\t%d\t\t%d\n" %
        (confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1])
    )

    # 对无标签数据进行分类
    for document in documents:
        if document.label is None:
            document.label = classifier.predict(document.words)

    # 保存搜索数据库
    pickle.dump(Database(
        documents,
        keywords
    ), db_file)

main()
