def karatsuba(x, y):
    # ベースケース
    if x < 10 or y < 10:
        return x * y

    # 数の桁数を決定
    n = max(len(str(x)), len(str(y)))
    m = n // 2

    # 分割
    # h 除算 l 剰余
    # ex)
    # x = 1234, y = 5678
    # n = 4
    # m = 2
    # x_l = 12, x_r = 34
    # y_l = 56, y_r = 78
    x_l, x_r = divmod(x, 10**m)
    y_l, y_r = divmod(y, 10**m)

    # 3つの乗算
    # 低い桁同士の乗算
    z0 = karatsuba(x_r, y_r)
    # 低い桁と高い桁の和同士の乗算
    z1 = karatsuba((x_r + x_l), (y_r + y_l))
    # 高い桁同士の乗算
    z2 = karatsuba(x_l, y_l)

    # 結果を再構成
    r1 = z2 * 10 ** (2 * m)
    # r2 = (x_l * y_r + x_r * y_l) * 10 ** m
    r2 = ((z1 - z2 - z0) * 10 ** m)
    r3 = z0
    return r1 + r2 + r3


# テスト
r = karatsuba(1234, 5678)

print(r)  # 7006652
