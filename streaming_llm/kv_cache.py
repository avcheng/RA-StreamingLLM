import io
import pickle
import sys

import chromadb
import redis
import torch

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
BATCH_SIZE = 32


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

        self.use_retrieval = use_retrieval
        if self.use_retrieval:
            self.redis_id = 0
            self.redis_client = self.reset_redis_connection()
            self.redis_client.ping()
            settings = chromadb.get_settings()
            settings.allow_reset = True
            self.chroma_client = chromadb.Client(settings=settings)
            self.collection = self.chroma_client.get_or_create_collection("kv_cache")

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

    # TODO Might not need this in the future
    def reset_redis_connection(self):
        return redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=False,
            health_check_interval=15,
            socket_keepalive=True,
        )

    def get_nearest_neighbors(self, prompt_embedding: torch.Tensor, n: int):
        """Returns the n nearest neighbors to the prompt embedding

        Args:
            prompt_embedding (torch.Tensor): embedding of the prompt
            n (int): number of nearest neighbors to return

        Returns:
            list: list kv caches of nearest neighbors
        """
        # Generate query to find n nearest neighbors
        results = self.collection.query(
            query_embeddings=[prompt_embedding.tolist()],
            n_results=n
        )

        return self.redis_client.mget(results["ids"][0])

    def add_to_kv_redis_cache(self, kv_cache, embedding: torch.Tensor):
        """Adds the kv cache and embedding to redis

        Args:
            kv_cache: list of k-v pairs
            embedding (torch.Tensor): embedding of the k-v pairs
        """
        # Store k-v pairs in redis
        stacked_cache = torch.stack([torch.stack((k, v)) for (k, v) in kv_cache])
        print(f"Shape of stacked_cache {stacked_cache.size()}")
        print(f"Dtype of stacked_cache {stacked_cache.dtype}")
        print(f"Size of stacked_cache: {stacked_cache.element_size() * stacked_cache.nelement()}")
        
        # Serialize the tensor to a bytestring
        bytestring_buffer = io.BytesIO()
        torch.save(stacked_cache.flatten().tolist(), bytestring_buffer)

        # Get the bytestring from the buffer
        bytestring = bytestring_buffer.getvalue()
        print(f"bytestring {len(bytestring)} bytes long...")
        self.redis_id += 1

        obj = pickle.dumps(stacked_cache.flatten())
        while True:
            try:
                # break
                # TODO Need to reduce memory requirements for serialized object because it is too big rn
                self.redis_client.set(self.redis_id, obj)
                break
            except redis.exceptions.ConnectionError:
                print("Redis connection error, resetting connection")
                self.redis_client = self.reset_redis_connection()

        # Add document embedding to index
        doc_emb = embedding.mean(dim=(0, 1)).flatten()
        self.collection.add([str(self.redis_id)], [doc_emb.tolist()])
        print(f"{self.collection.count()} documents in collection")

    def add_relevant_kv_to_cache(self, past_key_values, embeddings: torch.Tensor, n):
        """Adds the n nearest neighbors to the cache

        Args:
            prompt_embedding (np.array): embedding of the prompt
            n (int): number of nearest neighbors to return
        """
        if self.use_retrieval:
            prompt_embedding = embeddings.mean(dim=1).flatten()
            nearest_neighbors = self.get_nearest_neighbors(prompt_embedding, n)
            neighbor_tensors = []
            for neighbor in reversed(nearest_neighbors):
                # kv_cache = torch.frombuffer(neighbor, dtype=torch.float16)
                kv_cache = pickle.loads(neighbor)
                kv_cache = kv_cache.reshape(40, 2, 1, 40, -1, 128)
                neighbor_tensors.append(kv_cache)
                print(kv_cache.size())

            if neighbor_tensors:
                neighbor_tensors = torch.cat(neighbor_tensors, dim=-2)

                pkv = torch.stack([torch.stack((k, v)) for (k, v) in past_key_values])
                pkv = torch.cat([pkv[:, :, :, :, :4, :], neighbor_tensors, pkv[:, :, :, :, 4:, :]], dim=-2)
                print(f"Shape of pkv {pkv.size()}")
                past_key_values = map(lambda x: (x[0], x[1]), list(pkv))
        return list(past_key_values)

        # return [
        #     [
        #         torch.cat(
        #             [
        #                 self.k_slice(k, 0, self.start_size),
        #                 self.k_slice(
        #                     k, seq_len - self.recent_size + num_coming, seq_len
        #                 ),
        #             ],
        #             dim=self.k_seq_dim,
        #         ),
        #         torch.cat(
        #             [
        #                 self.v_slice(v, 0, self.start_size),
        #                 self.v_slice(
        #                     v, seq_len - self.recent_size + num_coming, seq_len
        #                 ),
        #             ],
        #             dim=self.v_seq_dim,
        #         ),
        #     ]
        #     for k, v in past_key_values
        # ]

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
        if seq_len + num_coming <= self.cache_size:
            print(f"Embedding size {past_emb.mean(dim=(0, 1)).size()}")
            return past_key_values

        if self.use_retrieval:
            # past_emb should have length equal to all the new tokens that were passed in
            num_new_tokens = past_emb.size(1)
            print(f"{num_new_tokens} new tokens")

            for batch in range(num_new_tokens // BATCH_SIZE):
                evicted_kv = [
                    [
                        self.k_slice(
                            k, -num_new_tokens + (batch * BATCH_SIZE), -num_new_tokens + ((batch + 1) * BATCH_SIZE)
                        ),
                        self.v_slice(
                            v, -num_new_tokens + (batch * BATCH_SIZE), -num_new_tokens + ((batch + 1) * BATCH_SIZE)
                        ),
                    ]
                    for k, v in past_key_values
                ]
                evicted_emb = past_emb[:, batch *
                                       BATCH_SIZE:(batch + 1) * BATCH_SIZE, :]
                self.add_to_kv_redis_cache(evicted_kv, evicted_emb)

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
    
    def clear_cache(self):
        self.redis_client.flushall()
        self.redis_client = self.reset_redis_connection()
        self.chroma_client.reset()
        self.collection = self.chroma_client.get_or_create_collection("kv_cache")
