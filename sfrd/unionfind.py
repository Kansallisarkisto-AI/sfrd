class UnionFind:
    """
    Union-find data structure. Some special cases:
     - Find and union operations will add the node if it does not already exist!
    """

    def __init__(self):
        self.p = {}
        self.sz = {}

    def add(self, x):
        """
        Add x
        """
        if x not in self.p:
            self.p[x] = x
            self.sz[x] = 1

    def find(self, x):
        """
        Find x (add if not existing)
        """
        self.add(x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        """
        Union a and b (add if not existing)
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]
