a = [4, 5, 6, 1, 2, 3]


def min_heapify(a, i):
    # i 根
    # l 左の子
    # r 右の子
    # m 最小値

    l = 2 * i + 1
    r = 2 * i + 2
    n = len(a) - 1
    m = i

    # 配列内である、左の子が根より大きい
    if l <= n and a[m] > a[l]:
        m = l
    # 配列内である、右の子が根より大きい
    if r <= n and a[m] > a[r]:
        m = r
    if m != i:
        a[i], a[m] = a[m], a[i]
        # 根が変わるので、根の子に対して再帰的に呼び出す
        min_heapify(a, m)


def insert(a, x):
    a.append(x)

    build_min_heap(a)


def delete(a, i):
    if len(a) <= 1:
        return a.pop()

    m = a[i]

    a[i] = a.pop()

    build_min_heap(a)

    return m


def build_min_heap(a):
    for i in reversed(range(len(a)//2)):
        min_heapify(a, i)


def heap_sort(a):
    a = a.copy()
    build_min_heap(a)
    s_a = []
    for _ in range(len(a)):
        # ヒープソートで先頭に最小値が来る
        # 先頭と末尾を交換
        a[0], a[-1] = a[-1], a[0]
        # 末尾（最小値）を取り出す
        s_a.append(a.pop())
        # ヒープソートを作る
        min_heapify(a, 0)

    return s_a


min_heapify(a, 0)
print(a)

build_min_heap(a)
print(a)

insert(a, 0)
print(a)

delete(a, 0)
print(a)

r = heap_sort(a)
print(r)
