from langchain.embeddings.bases import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriver


class RedundantFilterRetriever(BaseRetriver):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query):
        # calculate Embeddings for query string
        emb = self.embeddings.embed_query(query)
        # take Embeddings and feed into max_marginal_relevance_serach_by_vector
        self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb, lambda_mult=0.8)
        return []

    async def aget_relevant_documents(self):
        return []
