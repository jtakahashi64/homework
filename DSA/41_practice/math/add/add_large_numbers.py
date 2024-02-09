def add_large_numbers(a, b):
    # 最大長を取得し、両数を同じ長さにする
    max_length = max(len(a), len(b))
    a = a.zfill(max_length)
    b = b.zfill(max_length)

    carry = 0
    result = ''

    # 一番右のビットから加算を開始
    for i in range(max_length - 1, -1, -1):
        bit_a = int(a[i])
        bit_b = int(b[i])

        # 加算とキャリーの処理
        total = bit_a + bit_b + carry
        carry = total // 2

        # 結果を結合
        result = str(total % 2) + result

    # 最後のキャリーを追加
    if carry != 0:
        result = '1' + result

    return result


# テスト
a = '1010'  # 10 in binary
b = '1101'  # 13 in binary


r = add_large_numbers(a, b)

print(r)  # 23 in binary
