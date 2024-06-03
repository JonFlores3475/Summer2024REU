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