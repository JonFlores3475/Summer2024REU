# Defines a recursive function that calls itself if it isn't 1, adding the previous value to it and then printing out, so
# if the number is 6, it will return 1, 3, 6, 10, 15, 21

def tri_recursion(k):
    # Checks to see if k is greater than 0
    if k > 0:
        result = k + tri_recursion(k - 1) # Recursively calls it, adding the recursively-called value to the current value of k
        print(result) # Prints the result of what it is after
    else:
        result = 0 # Sets the result to 0 if k = 0
    return result

# Calls the recursive function
print("\n\nRecursion Example Results")
tri_recursion(6)