# Lets get into some more complicated things.

# In python, there are Lists, Tuples, Sets, and Dictionaries

# **List**

# This is how you create a list. Define the contents using square brackets.
# Lists are ordered, changeable, and allow duplicate items
# They are indexed, where the first item is [0], second item is [1], and so on

thislist = ["apple", "banana", "cherry"]
print(thislist)
len(thislist) # Returns the number of elements in the list

# Lists can also contain different data types
list1 = ["abc", 34, True, 40, "male"]

# You can use the list() constructor to construct a list

thislist = list(("apple", "banana", "cherry"))
print(thislist)

# Indexing on lists

print(thislist[1])     # Gets the element at the 2nd element (at index 1, being indexes start at 0)
print(thislist[-1])    # Gets the last item in the list (cherry)
print(thislist[0:2])   # Gets the items in a range (from 0 (inclusive) to 2 (exclusive))
print(thislist[:2])    # Gets all items until the specified index, excluding it
print(thislist[1:])    # Gets all items starting from index 1 to the end of the list
print(thislist[-3:-1]) # Gets all elements from the 3rd index from the back to (but not including) the last element

# Can use the in command to see if a certain element is in the list
if "apple" in thislist:
    print("Yes, 'apple' is in the fruits list")

# We won't get into the others for now

# **Loops**

# *While Loop*

i = 1
while i < 6:
    print(i)
    i += 1

# Just like other loops, it has similar context to the other languages.
# It also allows for break statements, continue statements
# One difference though is Python allows while loops to have else statements, so
# when the condition is no longer true, it will execute the code in the else block.

# *For Loops*

# Goes through the list and prints each fruit in the list
fruits = ["apple", "banana", "cherry"]
for x in fruits:
    print(x)

# Loops through the letters of banana
for x in "banana":
    print(x)

# These loops also use the break and continue statements
# You can also use the range() function to go through a sequence of numbers

# Goes through the numbers 0 - 6
for x in range(6):
    print(x)

# Goes through the numbers 2 - 6
for x in range(2, 6):
    print(x)

# Goes through the numbers 2 - 30, stepping by 3 (aka 2 --> 5 --> 8, etc.)
for x in range(2, 30, 3):
    print(x)

# For loops also allow for a else keyword where if the condition is not met, it will execute what is in the else block

# **Functions**

# This is the classic way to define a function
# Define the function
def my_function() :
    print("Hello from a function")

# Call the function
my_function()

# Above is the syntax for calling and defining functions.
# For parameters, it is the same thing only you don't have to specify the type! You just put in the name
# Define the function
def my_function(location):
    print("Hello from " + location)

# Call the function
my_function("California")
my_function("Idaho")
my_function("Oregon")

# You can also include * (arbitrary arguments) or ** (arbitrary keyword arguments)

# Arbitrary Function Example
def arbitrary_function(*kids):
    print("The youngest child is " + kids[2])

arbitrary_function("Emil", "Tobias", "Linus")

# Arbitrary Keyword Example
def arbitrary_keyword_function(**kid):
    print("His last name is " + kid["lname"])

arbitrary_keyword_function(fname = "Tobias", lname = "Refsnes")

# You can also set default values, where if they don't put in a parameter, it will set it automatically
def country_function(country = "Norway"):
    print("I am from " + country)

country_function("Sweden")
country_function("India")
country_function()
country_function("Brazil")

# **Classes/Object**

# Just like OOP languages, we can make Classes/Objects in Python!

# Creates a class
class MyClass:
    x = 5
# Creates an Object of said class
p1 = MyClass()
print(p1.x)

class Person:
    def __init__(self, age):
        self.age = age

# For this, refer to sample_class.py

# Classes have a __init__() function initializes the functions attributes
# and sets the values (think of this as a constructor in Java or a struct in C)

# Classes also have a __str__() function which allows you to format a class in
# string form, like to toString() method in java

# The self parameter is a reference to a current instance of the class, and is used
# to access variables that belongs to class. However, it doesn't have to be names self, it can be
# named anything!

# You can modify object properties just by calling them and changing the value.
# You can also delete properties on an object using the del keyword
person = Person(5)
person.age = 10
print(person.age)
del person.age
print(person.age) # Throws an error since it doesn't exist
 

