import random

class AliasSampler:
    def __init__(self, probs):
        n = len(probs)
        q = [p * n for p in probs]
        J = [0] * n
        smaller, larger = [], []

        for i, qi in enumerate(q):
            if qi < 1.0:
                smaller.append(i)
            else:
                larger.append(i)

        while smaller and larger:
            small = smaller.pop()
            large = larger.pop()
            J[small] = large
            q[large] = q[large] - (1.0 - q[small])
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large) 

        for leftover in smaller + larger:
            q[leftover] = 1.0
            J[leftover] = leftover

        self.J = J
        self.q = q

    def sample(self):
        n = len(self.J)
        i = random.randrange(n)
        return i if random.random() < self.q[i] else self.J[i]
