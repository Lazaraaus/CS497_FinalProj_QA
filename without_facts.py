import numpy as np
import pdb

# Dict to hold embeddings
embeddings_dict = {}
# Load GloVe Embeddings
filename = 'glove_embeddings/glove.6B.50d.txt'
with open(filename, 'r') as file:
    content = file.readlines()
    for line in content:
        content_line_split = line.split()
        token = content_line_split[0]
        embedding = content_line_split[1:-1]
        embedding = np.array(embedding, dtype=np.float64)
        embeddings_dict[token] = embedding
    file.close()
        
# Cos Sim
token_1_embed = embeddings_dict['the']
token_2_embed = embeddings_dict['their']
token_3_embed = embeddings_dict['food']
token_4_embed = embeddings_dict['xbox']
token_5_embed = embeddings_dict['google']


def cos_sim(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    cosine_sim = dot_product / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return cosine_sim



print(cos_sim(token_1_embed, token_2_embed))
print(cos_sim(token_2_embed, token_3_embed))
print(cos_sim(token_3_embed, token_4_embed))
print(cos_sim(token_4_embed, token_5_embed))
pdb.set_trace()


