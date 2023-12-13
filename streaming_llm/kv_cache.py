import pickle
import sys
import os

import torch

import pinecone
from dotenv import load_dotenv

load_dotenv()   

pinecone.init(      
	api_key=os.getenv('PINECONE_API_KEY'),      
	environment='gcp-starter'      
)      


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

VECTOR_SIZE = 5120
BATCH_SIZE = 64


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        use_retrieval=False,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.index = pinecone.Index('65940')

        self.use_retrieval = use_retrieval
        if self.use_retrieval:
            self.db_id = 0

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def add_to_kv_pinecone(self, kv_cache, embedding: torch.Tensor):
        """Adds the kv cache and embedding to redis

        Args:
            kv_cache: list of k-v pairs
            embedding (torch.Tensor): embedding of the k-v pairs
        """
        # Store k-v pairs in redis
        print(kv_cache)
        # print(f"{len(serialized)} bytes long...")
        self.db_id += 1
        self.index.upsert(
            id=self.db_id,
            values=embedding.tolist(),
            metadata={'tokens': kv_cache},
        )

    def add_relevant_kv_to_cache(self, past_key_values, embeddings: torch.Tensor, n):
        """Adds the n nearest neighbors to the cache

        Args:
            prompt_embedding (np.array): embedding of the prompt
            n (int): number of nearest neighbors to return
        """
        if self.use_retrieval:
            prompt_embedding = embeddings.mean(dim=1).flatten()
            nearest_neighbors = self.index.query(
                vector=prompt_embedding,
                top_k=n, 
                include_metadata=True
            )
            for neighbor in nearest_neighbors['matches']:
                kv_cache = neighbor['metadata']['tokens']
                cache_length = len(kv_cache)
                print(len(kv_cache))
                # make space for the new tokens
                past_key_values = self.evict_for_space(past_key_values, cache_length)
                # loop through the past key values and add the new tokens after the first self.start_size tokens
                for i, (k, v) in enumerate(past_key_values):
                    past_key_values[i][0] = torch.cat(
                        [
                            self.k_slice(k, 0, self.start_size),
                            torch.tensor(kv_cache[i][0]),
                            self.k_slice(k, self.start_size, self.recent_size),
                        ],
                        dim=self.k_seq_dim,
                    )
                    past_key_values[i][1] = torch.cat(
                        [
                            self.v_slice(v, 0, self.start_size),
                            torch.tensor(kv_cache[i][1]),
                            self.v_slice(v, self.start_size, self.recent_size),
                        ],
                        dim=self.v_seq_dim,
                    )
                    

        return past_key_values

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space_db(self, past_key_values, past_emb, num_coming):
        if past_key_values is None:
            return past_key_values
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        print(
            f"{len(past_key_values)} k-v pairs in cache with key vector of size {past_key_values[0][0].size()} with memory size {past_key_values[0][0].element_size() * past_key_values[0][0].nelement()}")
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        if self.use_retrieval:
            # past_emb should have length equal to all the new tokens that were passed in
            num_new_tokens = past_emb.size(1)
            print(f"{num_new_tokens} new tokens")

            for batch in range(num_new_tokens // BATCH_SIZE):
                evicted_kv = [
                    [
                        self.k_slice(
                            k, -num_new_tokens + batch * BATCH_SIZE, -
                            num_new_tokens + (batch + 1) * BATCH_SIZE
                        ).tolist(),
                        self.v_slice(
                            v, -num_new_tokens + batch *
                            BATCH_SIZE, num_new_tokens +
                            (batch + 1) * BATCH_SIZE
                        ).tolist(),
                    ]
                    for k, v in past_key_values
                ]
                evicted_emb = past_emb[:, batch *
                                       BATCH_SIZE:(batch + 1) * BATCH_SIZE, :]
                self.add_to_kv_pinecone(evicted_kv, evicted_emb)

        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
    
    def delete_index(self):
        pinecone.delete_index('65940')
