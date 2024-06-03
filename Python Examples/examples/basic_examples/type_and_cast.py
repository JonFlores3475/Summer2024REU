# Showcases how to use the type() function and what it can be used for as well as casting variables as different types

x = 25      # int
y = 0.84    # float 
z = 2.56    # float
print("Current Values and Types")
print(type(x), " ", x) # int    25
print(type(y), " ", y) # float  0.84
print(type(z), " ", z) # float  2.56

x = float(x) # Change to float 
y = int(y)   # Change to int
z = str(z)   # Change to str

print("\nCasted Values and Types")
print(type(x), " ", x) # float  25.0
print(type(y), " ", y) # int    0
print(type(z), " ", z) # str    "2.56"