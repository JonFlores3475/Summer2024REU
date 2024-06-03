# Defines a file with two different classes, one a child and one a parent. The child class (Student) implements the parent class (Person)
# saying that a student is-a type of person, but a person isn't a type of student. The child class also defines its own attributes that it 
# can define and use for it's own function.

# Parent class that is being defined
class Person:
    # Initializes the values
    def __init__(self, fname, lname):
        self.firstname = fname
        self.lastname = lname

    # Prints the first and last name of the person.
    def printname(self):
        print(self.firstname, self.lastname)

# Creates a person and calls the printname function
x = Person("Adam", "Crayton")
x.printname()

# Child class that is being defined that is a child of Person
class Student(Person):
    # Initializes the values
    def __init__(self, fname, lname, year):
        super().__init__(fname, lname) # Calls the super constructor of the parent class
        self.graduationyear = year # Defines its own attribute

    # Creates a method to print the name of the person as well as the year they graduate
    def __str__(self):
        super().printname()
        print(self.graduationyear)

# Creates a Student and calls the str function
y = Student("Adam", "Crayton", 2026)
y.__str__()