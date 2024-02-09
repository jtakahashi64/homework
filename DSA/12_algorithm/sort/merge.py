a = [4, 2, 3, 1, 6, 5]
n = len(a)

# def merge_sort(a, l, r):
#     if r - l <= 0:
#         return

#     m = (l + r) // 2

#     merge_sort(a, l,     m)
#     merge_sort(a, m + 1, r)

#     mid = m
#     L_i = l
#     R_i = m + 1

#     while L_i <= mid and R_i <= r:
#         if a[L_i] <= a[R_i]:
#             L_i += 1
#         else:
#             # Shift elements
#             index = R_i
#             value = a[R_i]

#             while index != L_i:
#                 a[index] = a[index - 1]
#                 index -= 1

#             a[L_i] = value

#             L_i += 1
#             R_i += 1
#             mid += 1

# merge_sort(a, 0, n - 1)

def merge_sort(a):
    if len(a) > 1:
        mid = len(a) // 2
        L = a[:mid]
        R = a[mid:]

        merge_sort(L)
        merge_sort(R)

        L_i = R_i = a_i = 0

        while L_i < len(L) and R_i < len(R):
            if L[L_i] < R[R_i]:
                a[a_i] = L[L_i]
                L_i += 1
            else:
                a[a_i] = R[R_i]
                R_i += 1
            a_i += 1

        # Copy remaining elements
        while L_i < len(L):
            a[a_i] = L[L_i]
            L_i += 1
            a_i += 1

        # Copy remaining elements
        while R_i < len(R):
            a[a_i] = R[R_i]
            R_i += 1
            a_i += 1

        return a

a = merge_sort(a)

print(a)
