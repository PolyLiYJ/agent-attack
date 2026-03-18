import os
import json

folder = "/home/yjli/Agent/agent-attack/cache/success--results_20250708182630_gpt-4o-mini_typography_attack_0/action_results"  # e.g., 'memory_bank/'
memory_records = []
file_paths = [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.endswith(".json")]

for fp in file_paths:
    with open(fp, "r") as f:
        rec = json.load(f)
        memory_records.append(rec)
        
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
client = OpenAI()

# embeddings = []
# prompts = [rec['injected_prompt'] for rec in memory_records]
# for prompt in prompts:
#     response = client.embeddings.create(
#     input=prompt,
#     model="text-embedding-3-small"
#     )
#     embeddings.append(response.data[0].embedding)

#embeddings = np.stack(embeddings)
#np.save(os.path.join("/home/yjli/Agent/agent-attack/cache/success--results_20250708182630_gpt-4o-mini_typography_attack_0/", 'embeddings.npy'), embeddings)
embeddings = np.load(os.path.join("/home/yjli/Agent/agent-attack/cache/success--results_20250708182630_gpt-4o-mini_typography_attack_0/", 'embeddings.npy'))


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve_similar_few_shots(new_injected_prompt, topk=3, score_threshold=0.8):
    """
    Retrieves top-k most similar few-shot examples from memory based on cosine similarity.
    
    Args:
        new_injected_prompt (str): The input prompt to find similar examples for.
        model: The embedding model used to encode the prompt.
        topk (int, optional): Number of similar examples to return. Defaults to 3.
        score_threshold (float, optional): Minimum similarity score threshold. Defaults to 0.8.
    
    Returns:
        list: Top-k most similar memory records that meet the score threshold.
    """
    query_vec = client.embeddings.create(
    input=new_injected_prompt,
    model="text-embedding-3-small"
    ).data[0].embedding
    query_vec = [np.array(query_vec)]
    sims = cosine_similarity(query_vec, embeddings)[0]
    # Optionally only return successful examples, e.g. score > 0.0
    candidates = [(rec, sim) for rec, sim in zip(memory_records, sims) if rec['score'] > score_threshold]
    candidates = sorted(candidates, key=lambda x: -x[1])
    return [rec for rec, sim in candidates[:topk]]


# print(retrieve_similar_few_shots("what is the Email"))