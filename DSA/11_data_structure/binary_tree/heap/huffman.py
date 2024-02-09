from heapq import heappush, heappop, heapify


# Define the function to create Huffman tree
def create_huffman_tree(frequencies):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]

    heapify(heap)

    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)

        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]

        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))


# Frequencies of the letters as given in the problem
frequencies = {
    'A': 35,
    'B': 25,
    'C': 20,
    'D': 15,
    'E': 5
}

# Create the Huffman Tree
r = create_huffman_tree(frequencies)

print(r)
