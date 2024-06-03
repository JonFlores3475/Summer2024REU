# Some things in this language can be a little complicated and a little hard to
# understand. Let's get into some of them.

# **Lambda**

# Lambda unctions are small anonymous functions with the syntax of:
    # lambda arguments : expression

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

# **Inheritance and Polymorphism**

# Python has both inheritance and polymorphism built into the language. For inheritance,
# you can create multiple classes where one "extends"/"implements" the other by having the 
# other class being the parameter of the second class.

class Example1():
    def __init__(self, fname, lname):
        self.firstname = fname
        self.lastname = lname

class Example2(Example1):
    pass

# In the case, Example1 is a class we made. Example2 is a class we made that inherits all aspects
# of Example1.
# Example2 can then implement their own __init__. Example2 can call example1's __init__ function in
# multiple ways as shown below:

class Example3(Example1):
    def __init__(self, fname, lname):
        Example1.__init__(self, fname, lname)

    def __init__(self, fname, lname):
        super().__init__(fname, lname)

# You can then add your own attribtes to that class just like how you did with the child class! 
# Refer to inheritence.py for more information


# Python allows the program to call multiple objects with the name method names where it will execute each
# method individually, as shown below

class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def move(self):
        print("Drive!")

class Boat:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def move(self):
        print("Sail!")

class Plane:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def move(self):
        print("Fly!")

car1 = Car("Ford", "Mustang") # Create a Car class
boat1 = Boat("Ibiza", "Touring 20") # Create a Boat class
plane1 = Plane("Boeing", "747") # Create a Plane class

for x in (car1, boat1, plane1):
    x.move()


# Polymorphism and Inheritance can also be combined together.
# Reference polymorphism.py to see what we mean

# **Modules**

# Modules are basically libraries in Python. It is a file containing a set of functions to include in
# your application

# Refer to the module_examples folder for more information (check out main.py)

