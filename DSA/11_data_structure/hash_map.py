class HashMap:
    def __init__(self, size, threshold):
        self.size = size
        self.threshold = threshold
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0

    def hash_function(self, key):
        return hash(key) % len(self.buckets)

    def resize(self):
        if self.count / self.size <= self.threshold:
            return

        self.size *= 2
        new_buckets = [[] for _ in range(self.size)]

        for bucket in self.buckets:
            for key, value in bucket:
                index = self.hash_function(key)
                new_buckets[index].append([key, value])
            self.buckets = new_buckets

    def insert(self, key, value):
        self.resize()
        index = self.hash_function(key)

        for item in self.buckets[index]:
            if item[0] == key:
                item[1] = value
                return

        self.buckets[index].append([key, value])
        self.count += 1

    def search(self, key):
        index = self.hash_function(key)

        for item in self.buckets[index]:
            if item[0] == key:
                return item[1]

        return None

    def delete(self, key):
        index = self.hash_function(key)

        for i, item in self.buckets[index]:
            if item[0] == key:
                del self.buckets[index][i]
                return True

        return False


h = HashMap(5, 0.7)

h.insert('a', 1)
h.insert('b', 2)
h.insert('c', 3)

print(h.buckets)

h.insert('d', 4)
h.insert('e', 5)
h.insert('f', 6)

print(h.buckets)

r = h.search('a')
print(r)
