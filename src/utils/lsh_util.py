import numpy as np
from init import embed_model

class RandomProjectionLSH:
    def __init__(self, dim, num_bits):
        self.planes = np.random.randn(num_bits, dim)  # 随机超平面法向量

    def hash_vector(self, vec):
        projections = np.dot(self.planes, vec)
        return (projections > 0).astype(int)

    def get_bucket_id(self, query_vector):
        binary_hash = self.hash_vector(query_vector)
        # 转为字符串作为桶 ID
        return ''.join(map(str, binary_hash))

