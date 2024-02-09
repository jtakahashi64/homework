# k番目に小さい要素を求める
def quick_select(a, k):
    if len(a) <= 1:
        return a[0]

    pivot = a[0]

    L = [x for x in a if x <= pivot and x != pivot]
    R = [x for x in a if x > pivot and x != pivot]

    # k番目に小さい要素がpivotになるようにする
    if len(L) == k - 1:
        return pivot

    # k番目に小さい要素がpivotより左側にある場合
    if len(L) > k - 1:
        return quick_select(L, k)
    # k番目に小さい要素がpivotより右側にある場合
    else:
        return quick_select(R, k - len(L) - 1)


a = [4, 2, 3, 1, 6, 5]

r = quick_select(a, 3)

print(r)
