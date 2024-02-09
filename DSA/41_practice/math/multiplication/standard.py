def standard_multiplication(x, y):
    result = 0
    place = 0

    while x > 0:
        # 一番右の桁を取得
        d = x % 10
        # 一番右の桁を削除
        x = x // 10

        temp_result = d * y

        # 桁に応じて結果をずらす
        temp_result = temp_result * (10 ** place)
        place += 1

        # 結果を加算
        result += temp_result

    return result


r = standard_multiplication(123, 456)

print(r)  # 56088
