# Lambda unctions are small anonymous functions with the syntax of -- lambda arguments : expression

# Example:
x = lambda a : a + 10
print(x(5)) # This will return 15 as 5 + 10 = 15

# Lambdas can also have multiple arguments to be able to use
x = lambda b, c : b * c
print(x(5, 6)) # This will return 30 since 5 * 6 = 30

# Lambda functions are useful in regular function calls. It allows us to call them inside the function
# and to be used to define singular expressions for a function

def myfunc(n):
    return lambda a : a * n

mydoubler = myfunc(2)
mytripler = myfunc(3)
print(mydoubler(11))
print(mytripler(33))