import chromadb
from typing import List

class VectorStoreUtils: 

    def __init__(self):
        self.chromaClient = chromadb.EphemeralClient() 
        self.collection = chroma_client.create_collection(name="eustachian_collection")
        self.lastId = 0
        
    def addToCollection(self, documents: List): 
        for i in documents:
         entry =  f"""
        {{ 
            "caption": "{i.caption}",
            "base64": "{self._encodeImage(f'{i.image}')}"
        }}
        """
        self.collection.add(
           ids = [self.lastId],
           documents = [
              entry 
           ]
        )
        self.lastId += 1      