
def myfunc1():
    x = 300
    print(x)

myfunc1()
# print(x) --> Will not print because "x" does not exist in the scope


def myfunc2():
    x = 300
    def myinnerfunc():
        print(x)
    myinnerfunc()

myfunc2()

# This will print x though because it is in an "outer" scope, allowing the inner function to call it

y = 300

def myfunc4():
    print(y)

myfunc4()

print(y)

# Being that y is a global variable, it can be accessed anywhere, in or outside a method

z = 300

def myfunc5():
    global z
    z = 200

myfunc5()

print(z)

# The 'global' keyword allows a function to access the element and make changes to it globally rather than locally in a scope

z = 300

def myfunc6():
    z = 200

myfunc6()

print(z)

# You need to use the 'global' keyword in order to actually modify the value