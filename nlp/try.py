a = [[1, 2, 3], [4, 5, 6]]

b = [v for line in a for v in line]
# print(b,'\n', a)
# 妙！
c = lambda x: x ** 2
d = [1, 2, 3]
g = d[:10]
print(g)
f = [1,[2, 3, [4, 5]]]
e = [4] + d
# print(e)