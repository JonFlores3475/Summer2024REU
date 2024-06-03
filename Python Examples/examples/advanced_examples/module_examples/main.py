# This is showing the basics of how to create a module and import is,
# being able to use the methods and functions inside that file. (look at greeting.py)
import greeting

greeting.greeting("Adam")

# This shows how you can even access variables in a module. When we
# import another file, it gives us access to everything in that file, including variables.

import person

a = person.person1["age"]
print(a)

# You can also rename the import and import it in anyway you want

import person as p

a = p.person1["age"]
print(a)

# Python also has multiple build in modules. Some examples include:
#   - platform
#   - os
#   - dir

# You can also oly import specific things from a module if you don't want the whole file.
# Below shows what it looks like when you only import the dictionary and not the method.

from person_hello import person1

print(person1["age"])
