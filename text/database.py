from typing import List

class KeywordOccurrenceInDocument:
    def __init__(self, document_id: int, positions: List[int]):
        self.document_id = document_id
        self.tf: float = 0
        self.positions = positions

class Keyword:
    def __init__(self, word: str):
        self.word = word
        self.occurs: List[KeywordOccurrenceInDocument] = []
        self.idf: float = 0

class Document:
    def __init__(self, title: str, content: str, words: List[str], label: bool):
        self.title = title
        self.content = content
        self.words = words
        self.label = label

class Database:
    def __init__(self, documents: List[Document], keywords: List[Keyword]):
        self.documents = documents
        self.keywords = keywords
