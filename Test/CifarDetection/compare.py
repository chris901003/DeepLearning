file1 = '410985015.txt'
file2 = '410985048.txt'

with open(file1, 'r') as f:
    info1 = f.readlines()

with open(file2, 'r') as f:
    info2 = f.readlines()

res = 0
for idx in range(len(info1)):
   if info1[idx] == info2[idx]:
       res += 1
print(res)
