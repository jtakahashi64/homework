a = [1, 2, 3, 4, 5, 6]
n = len(a)

# def binary_search(a, x):
#     l = 0
#     r = len(a) - 1
#     m = 0

#     while l <= r:
#         m = (l + r) // 2
#         if a[m] < x:
#             l = m + 1
#         if a[m] > x:
#             r = m - 1
#         if a[m] == x:
#             return m

#     return -1


def binary_search_recursive(a, x, l, r):
    if l > r:
        return -1

    m = (l + r) // 2

    if a[m] < x:
        return binary_search_recursive(a, x, m + 1, r)
    if a[m] > x:
        return binary_search_recursive(a, x, l, m - 1)
    if a[m] == x:
        return m


r = binary_search_recursive(a, 3, 0, n - 1)

print(r)
