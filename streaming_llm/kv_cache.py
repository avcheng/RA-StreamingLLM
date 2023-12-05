import pickle

import redis
import torch
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query


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

VECTOR_SIZE = 64


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.redis_id = 0
        self.redis_client = redis.Redis(
            host='localhost', port=6379, decode_responses=True)

        schema = (
            VectorField(
                "embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_SIZE,
                    "DISTANCE_METRIC": "COSINE",
                }
            ),
            TagField("kv_id")
        )
        idx_def = IndexDefinition(
            prefix=["key:"], index_type=IndexType.HASH)
        self.index = "idx"
        self.redis_client.ft(self.index).create_index(
            schema, definition=idx_def)

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

    def get_nearest_neighbors(self, prompt_embedding, n):
        """Returns the n nearest neighbors to the prompt embedding

        Args:
            prompt_embedding (np.array): embedding of the prompt
            n (int): number of nearest neighbors to return

        Returns:
            list: list kv caches of nearest neighbors
        """
        # Generate query to find n nearest neighbors
        query = (
            Query(f"(*)=>[KNN {n} @embedding $query_vector AS vector_score]")
            .sort_by("vector_score")
            .paging(0, n)
            .return_fields("kv_id")
            .dialect(2)
        )

        result_docs = self.redis_client.ft(self.index).search(
            query,
            query_params={"query_vector": prompt_embedding.tolist()},
        ).docs

        return [self.redis_client.get(doc.kv_id) for doc in result_docs]

    def add_to_kv_redis_cache(self, kv_cache, embedding: torch.Tensor):
        """Adds the kv cache and embedding to redis

        Args:
            kv_cache (list): list of k-v pairs
            embedding (torch.Tensor): embedding of the k-v pairs
        """
        # Store k-v pairs in redis
        serialized = pickle.dump(kv_cache)
        self.redis_id += 1
        self.redis_client.set(self.redis_id, serialized)

        # Add document embedding to index
        self.redis_client.ft(self.index).add_document(
            self.redis_id, {"embedding": embedding.tolist(), "kv_id": self.redis_id})

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
            return past_key_values, past_emb
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values, past_emb

        # TODO Figure out which k-v pairs should be evicted
        # TODO Figure out way to batch evicted tokens into batches of a certain (TBD) size
        # TODO Figure out way to identify which tokens/k-v pairs have already been saved to redis
        # grab evicted tokens and k-v pairs for reuse
        evicted_kv = [
            [
                self.k_slice(
                    k, self.start_size, seq_len - self.recent_size + num_coming
                ),
                self.v_slice(
                    v, self.start_size, seq_len - self.recent_size + num_coming
                ),
            ]
            for k, v in past_key_values
        ]
        evicted_emb = past_emb[:seq_len + num_coming - self.cache_size]
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
        # TODO: grab evicted token k-v pairs for reuse
        # evicted = [
        #     [
        #         self.k_slice(k, start, end),
        #         self.v_slice(v, start, end),
        #     ]
        #     for k, v in past_key_values
        # ]
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
