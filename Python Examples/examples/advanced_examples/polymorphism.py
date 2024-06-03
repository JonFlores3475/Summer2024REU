# Creates a class showing polymorphism (and inheritance) and due to the built in polymorphism of Python, it allows us to call methods of different
# classes even if they may have the same name and different functionalities.

# Defines the parents class of the vehicle
class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model 
    def move(self):
        print("Move!")

# Defines a child-Car class that does everything vehicle does and nothing more
class Car(Vehicle):
    pass

# Defines a child-Boat class that changes the value of the print statement to "Sail!" what move() is called
class Boat(Vehicle):
    def move(self):
        print("Sail!")

# Defines a child-Plane class that changes the value of the print statement to "Fly!" what move() is called
class Plane(Vehicle):
    def move(self):
        print("Fly!")

car1 = Car("Ford", "Mustang") # Create a Car object
boat1 = Boat("Ibiza", "Touring 20") # Create a Boat object
plane1 = Plane("Boeing", "747") # Create a Plane object

# Goes through each object, printing out their brand, model, and calling the move() method (printing "Move!", "Sail!", and "Fly!")
for x in (car1, boat1, plane1):
    print(x.brand)
    print(x.model)
    x.move()