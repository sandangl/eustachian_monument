import chromadb
import base64
import io
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from typing import List

class VectorStoreUtils: 

    def __init__(self):
        self.chromaClient = chromadb.EphemeralClient() 
        self.collection = self.chromaClient.create_collection(name="eustachian_collection")
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

    def _encode_image(self, image_path) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _decode_and_show_image(self, base64_img, img_format: str):
        decoded_bytes = io.BytesIO(base64.b64decode(base64_img))
        decoded_image = mpimg.imread(decoded_bytes, format=img_format)
        
        plt.imshow(decoded_image, interpolation='nearest')
        plt.show()  