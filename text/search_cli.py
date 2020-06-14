import pickle
import argparse
import statistics
import readline
import time
from typing import List

import termcolor

start_time = time.time()

print('Loading jieba...')

import utility
from database import Database, Document, Keyword, KeywordOccurrenceInDocument

parser = argparse.ArgumentParser()
parser.add_argument("database_file", help="The path of file to store the database")
args = parser.parse_args()

# 加载数据库
print('Loading index database...')

database: Database = pickle.load(open(args.database_file, "rb"))

print('\nInitizlized in %s second(s).\n' % termcolor.colored("%.3lf" % (time.time() - start_time), color="green", attrs=["bold"]))

# 对给定的关键词进行搜索
def do_search(search_keywords: List[str]):
    if database.keywords.get(search_keywords[0]) == None:
        return []

    # 检查之后的单词是否匹配
    def check_following_words(keyword_occur: KeywordOccurrenceInDocument):
        document = database.documents[keyword_occur.document_id]
        for pos in keyword_occur.positions:
            mismatch = False
            try:
                for i in range(0, len(search_keywords)):
                    if search_keywords[i] != document.words[pos + i]:
                        mismatch = True
                        break
            except IndexError:
                mismatch = True

            if not mismatch:
                return True
        return False
    # 从倒排索引中取出文档列表，并检查其每次出现的之后的单词是否匹配，并筛选出文档 ID
    occurs = filter(check_following_words, database.keywords[search_keywords[0]].occurs)
    document_ids = map(lambda occur: occur.document_id, occurs)

    weight = {}
    for document_id in document_ids:
        weight[document_id] = 0
    
    # 通过 TF-IDF 对搜索结果进行排序
    for word in search_keywords:
        for occur in database.keywords[word].occurs:
            try:
                weight[occur.document_id] += occur.tf * database.keywords[word].idf
            except KeyError:
                pass

    results = list(zip(weight.keys(), weight.values()))
    results.sort(key=lambda x: x[1], reverse=True)

    return results

# 处理输入，进行切词与搜索
def process_input(input: str):
    search_keywords = utility.cut_words(input)

    print("\nKeywords to search: %s\n" % ' '.join([
        termcolor.colored(word, color="yellow", attrs=["bold"])
        for word in search_keywords
    ]))

    results = do_search(search_keywords)

    # 输出搜索结果
    print("%s result(s) found.\n" % termcolor.colored(len(results), color="green", attrs=["bold"]))

    trash = termcolor.colored("Trash", color="red", attrs=["bold"])
    non_trash = termcolor.colored("Non-Trash", color="green", attrs=["bold"])
    for i in range(len(results)):
        result = results[i]
        document = database.documents[result[0]]
        print("%s (TF-IDF weight = %s): %s [%s]\n%s" % (
            termcolor.colored("Result #%d" % (i + 1), color="green", attrs=["bold"]),
            termcolor.colored("%.6lf" % result[1], attrs=["bold"]),
            termcolor.colored(document.title, color=("blue" if document.title.startswith("Labeled") else "magenta"), attrs=["bold"]),
            trash if document.label else non_trash,
            document.content
        ))
        print('')

# 实现一个 REPL
try:
    while True:
        try:
            input_keywords = input('Search> ').strip()
            if not input_keywords:
                continue

            start_time = time.time()
            process_input(input_keywords)
            end_time = time.time()

            print('Query finished in %s second(s).\n' % termcolor.colored('%.6lf' % (end_time - start_time), color="green", attrs=["bold"]))
        except KeyboardInterrupt:
            print('')
            continue
except EOFError:
    print('')
