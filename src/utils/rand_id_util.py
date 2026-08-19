import time
import random

class Random:
    uid: int

    @staticmethod
    def gen_simple_inc_random() -> int:
        ts = int(time.time() // 10000) * 10000
        rand_suffix = random.randint(0, 999)
        uid = ts + rand_suffix
        return uid

