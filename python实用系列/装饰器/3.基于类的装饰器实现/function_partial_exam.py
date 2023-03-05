import functools
def add_number_one(a):
	return a+1
add=functools.partial(add_number_one,10)
print(add())
# 11
