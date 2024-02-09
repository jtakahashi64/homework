a = [4, 2, 3, 1, 6, 5]

def quick_sort(a):
    if len(a) <= 1:
        return a
    else:
        pivot = a[0]
        L = [x for x in a if x <= pivot and x != pivot]
        R = [x for x in a if x >  pivot and x != pivot]
        return quick_sort(L) + [pivot] + quick_sort(R)

a = quick_sort(a)

print(a)
