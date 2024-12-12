# Go to the following link for a full tutorial: https://docs.python.org/3/tutorial/
import math
from collections import defaultdict 
import os
import csv
#Basic Printing
print('Hello World')

# Assignment
i=0
j=1
k=2
l="Hello"
h='Goodbye'
w=True
x=1.5

#Data Types
type(i)
type(l)
type(w)
type(x)
float(j)
int(x)
int(w)
int(h)

#Python is case sensitive
"Hello"=="Hello"
"Hello"=="hello"
1=true
1==true
1==True
1==False


#Addition and subtraction
k+j
l+h
j+l
j+w
l+w

#Multiplication and division
k*j
k/j
k/i
5*15
5/15
15/5
w*j
l*w
l*10

#Modulo
53%10
53%9

#Exponential
1**0
2**2
5**5

#Comparison

#Logical Operators
True or False
True | False
True and False
True & False
(5==6) or (5*6==30)
(5==6) and (5*6==30)


#For loops
for i in range(10):
	print(i)

#While loops
i=0
while i<10:
	print(i)
	i=i+1

#Functions
def HelloWorld(i):
	for j in range(i):
		print('Hello World')

#If statements
def HelloWorld(i):
	for j in range(i):
		if i>=5:
			print('Hello World')

#Return Statements
def Square(input):
	return input**2

Square(4)
Square(5)

def isItASquare(input):
	if int(math.sqrt(input))-math.sqrt(input)==0:
		return True
	else:
		return False

isItASquare(4)

isItASquare(5)

isItASquare(Square(5))

#Data Structures: Lists

listofNames=[]
listofNames.append('Harry')
listofNames2=['Sally','Pokemon','batman','Bibi','Sally']
print(listofNames2)
listofNames3 = listofNames2 + listofNames
print(listofNames3)
listofNames.extend(listofNames2)
print(listofNames)
listofNames.remove(Pokemon)
listofNames.remove('pokemon')
listofNames.remove('Pokemon')
print(listofNames)

#Data Structures: Tuples

thistuple = ("apple", "banana", "cherry")
print(thistuple)
thistuple = ("apple", "banana", "cherry", "apple", "cherry")
print(thistuple)
print(len(thistuple))

#Data Structures: Dictionaries

thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(thisdict)

print(thisdict["brand"])

thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964,
  "year": 2020
}
print(thisdict)

thisdict = dict(name = "John", age = 36, country = "Norway")
print(thisdict)

#Default Dictionaries

def def_value(): 
    return "Not Present"

d = defaultdict(def_value) 
d["a"] = 1
d["b"] = 2
  
print(d["a"]) 
print(d["b"]) 
print(d["c"]) 

d = defaultdict(int) 
L = [1, 2, 3, 4, 2, 4, 1, 2] 
for i in L:
	d[i] += 1

print(d)

#Sets

thisset = {"apple", "banana", "cherry"}
print(thisset)

thisset = {"apple", "banana", "cherry", "apple"}

print(thisset)

thisset = {"apple", "banana", "cherry", True, 1, 2}

print(thisset)

print(len(thisset))

myset = {"apple", "banana", "cherry"}
print(type(myset))

#Recursion
def factorial(n):
	if n==0:
		return 1
	else:
		return n*factorial(n-1)

factorial(10)

def fibonacci(n):
	if n==1:
		return 1
	elif n==0:
		return 1
	else:
		return fibonacci(n-1)+fibonacci(n-2)

#Dynamic Programming
def factorialDynamicProgramming(n):
	fDict={}
	fDict[0]=1
	for i in range(1,n+1):
		fDict[i]=fDict[i-1]*i
	return fDict[n]

factorialDynamicProgramming(5)

def fibonacciDynamicProgramming(n):
	fDict={}
	fDict[0]=0
	fDict[1]=1
	for i in range(2,n+1):
		fDict[i]=fDict[i-1]+fDict[i-2]
	return fDict[n]

fibonacciDynamicProgramming(8)

#File i/O
os.getcwd()
os.chdir('yourWorkingDirectory')
with open('data.csv', newline='') as csvfile:
	data = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in spamreader:
		print(', '.join(row))

with open('names.csv', 'w', newline='') as csvfile:
    fieldnames = ['first_name', 'last_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
    writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
    writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})

#Error Handling

def divideXByY(x,y):
	try:
		return x/y
	except:
		return "Can't be done!"

for i in range(10):
	for j in range(0,20,2):
		divideXByY(i,j)

#Classes

class Car:
    # Class attribute
    car_count = 0
    
    # Constructor (initializer method)
    def __init__(self, brand, model, year):
        # Instance attributes
        self.brand = brand
        self.model = model
        self.year = year
        Car.car_count += 1  # Increment the car count
    
    # Instance method
    def display_info(self):
        print(f"{self.brand} {self.model} ({self.year})")
    
    # Another instance method
    def drive(self):
        print(f"{self.brand} {self.model} is driving.")

#Create instances

car1 = Car("Toyota", "Corolla", 2020)
car2 = Car("Honda", "Civic", 2018)
car3 = Car("Ford", "Mustang", 2022)

# Call instance methods
car1.display_info()  # Output: Toyota Corolla (2020)
car2.drive()         # Output: Honda Civic is driving.

print(f"Car 1: {car1.brand}, {car1.model}, {car1.year}")

print(f"Total Cars: {Car.car_count}")

# Class Inheritence
class ElectricCar(Car):
    def __init__(self, brand, model, year, battery_capacity):
        # Call the constructor of the parent class (Car)
        super().__init__(brand, model, year)
        # Additional attribute specific to ElectricCar
        self.battery_capacity = battery_capacity
    
    # Override drive method
    def drive(self):
        print(f"{self.brand} {self.model} is driving silently with {self.battery_capacity} kWh battery.")

electric_car = ElectricCar("Tesla", "Model S", 2023, 100)
electric_car.display_info()  # Output: Tesla Model S (2023)
electric_car.drive()