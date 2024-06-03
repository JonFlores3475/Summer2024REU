
import sys

# This program prints Hello, world! And the version of Python we have. The print function automatically adds a newline character at the end

print('Hello, world!')
print(sys.version)

# If an if statement or anything that involves you to indent doesn't have an indentation on the next line, 
# python will not allow you to run it if it isn't indented

# Will throw error due to no indentation
'if 5 > 2:'
'print("5 is greater than two!")'

# Correct version
if 5 > 2:
    print('5 is greater than two')

# Variables can be anything! You don't have to specify what kind it is.

x = 2
y = "Hello there!"
z = True

# These serve as comments
' These funny enough also serve as comments! '

# Here are some casting examples

x = float(3) # This will be 3.0
y = str(3)   # This will be '3'

# You can get the data type of a variable with the type() method

print(type(x))
print(type(y))

# Strings can be denoted with both '' and "", both will work
# Variables are also case sensitive, meaning a and A will be different

# Variable Name Rules:
#   - Must start with a letter or '_'
#   - Cannot start with a number
#   - Can only contain alpha-numeric characters and underscores
#   - Cannot be any of the Python keywords (https://www.w3schools.com/python/python_ref_keywords.asp)

# Valid
myvar = "John"
my_var = "John"
_my_var = "John"
myvar = "John"
MYVAR = "John"
myvar2 = "John"

# Invalid
'2myvar = "John"'
'my-var = "John"'
'my var = "John"'