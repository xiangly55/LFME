

path = 'ImageNet_LT_test_bak.txt'
path_new = 'ImageNet_LT_test.txt'
f = open(path, 'r')
f_new = open(path_new, 'a')


for line in f.readlines():
	li = line.split('/')
	li.remove(li[1])
	f_new.write('/'.join(li))

f.close()
f_new.close()
