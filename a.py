from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passsage')

doc = searcher.doc('7157715')
#print(doc)
data = []

for i in range(searcher.num_docs):
    data.append(searcher.doc(str(i)))

print(len(data))