# Here is a sample program of a class with the __init__ and __str__ functions
# It defines a person who has a name and age associated with them
# It also shows an in-class function that makes the age

class Person:
    # Initializes the person, setting the names and ages
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Gains the information and prints it
    def __str__(information):
        return f"{information.name}({information.age})"
    
    # Increments the persons age by one and gives them a happy birthday message
    def birthday(self):
        self.age = self.age + 1
        print("It's your birthday! Happy birthday! You are now " + str(self.age) + " years old")

# Sample of creating an object, executing the method, and printing out the results
p1 = Person("Adam", 20)
p1.birthday()
p1.__str__()