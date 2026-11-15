
import tiktoken
import numpy as np
from openai import OpenAI  
from dotenv import load_dotenv



load_dotenv()  # Load environment variables from .env file

TEST_PAIRS = [
    ("success", "sucsses"),
    ("definitely", "definately"),
    ("embedding", "embeding"),
    ("tokenization", "tokenizaton"),
]

BASELINE = "banana"

def show_tokens(enc, word):
    ids = enc.encode(word) 
    #this split the word into tokens and returns a list of the 
    #ids using the encode function from OpenAI's tiktoken library

    subparts_list = [] #store the subparts of the word in a list
    for id in ids:
        subpart = enc.decode([id]) #decodes each id to its coressponding subpart of the word
        subparts_list.append(subpart)

    return len(ids), subparts_list
        


def embed(words):
    api = OpenAI() #client for OpenAI API
    embeddings = {} # stores the embedding values for each word
    for word in words:
        response = api.embeddings.create(
            model="text-embedding-3-small",
            input=word,
        ) #the responce from API that contains the embedding values
        embeddings[word] = np.array(response.data[0].embedding) #stores the embedding values in a numpy array each containing around 1500 values for each word
    return embeddings



def cosine(a, b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)) #this function calculates the cos similiarity between two vectors 


def main():
    enc = tiktoken.get_encoding("o200k_base")
    for w1, w2 in TEST_PAIRS:
        count1, pieces1 = show_tokens(enc, w1)
        count2, pieces2 = show_tokens(enc, w2)
        print(f"Word: {w1}, Token Count: {count1}, Pieces: {pieces1}")
        print(f"Word: {w2}, Token Count: {count2}, Pieces: {pieces2}")
        print()

    

   

    for w1, w2 in TEST_PAIRS:

        embeddings = embed([w1, w2, BASELINE])
        similarity = cosine(embeddings[w1], embeddings[w2])
        baseline_similarity = cosine(embeddings[w1], embeddings[BASELINE])
        print(f"Cosine similarity between '{w1}' and '{w2}': {similarity:.4f}")
        print(f"Cosine similarity between '{w1}' and '{BASELINE}': {baseline_similarity:.4f}")
        print()



if __name__ == "__main__":
    main()
