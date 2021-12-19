(define (problem ztravel-5-4)
(:domain zeno-travel)
(:objects
plane1 - aircraft
plane2 - aircraft
plane3 - aircraft
plane4 - aircraft
plane5 - aircraft
person1 - person
person2 - person
person3 - person
person4 - person
city0 - city
city1 - city
city2 - city
city3 - city
city4 - city
city5 - city
fl0 - flevel
fl1 - flevel
fl2 - flevel
fl3 - flevel
fl4 - flevel
fl5 - flevel
fl6 - flevel
)
	(:init
			(at person1 city3)
			(at person2 city5)
			(at person3 city1)
			(at person4 city2)
			(at plane1 city5)
			(at plane2 city0)
			(at plane3 city5)
			(at plane4 city0)
			(at plane5 city0)
			(fuel-level plane1 fl0)
			(fuel-level plane2 fl0)
			(fuel-level plane3 fl2)
			(fuel-level plane4 fl1)
			(fuel-level plane5 fl5)
			(next fl0 fl1)
			(next fl1 fl2)
			(next fl2 fl3)
			(next fl3 fl4)
			(next fl4 fl5)
			(next fl5 fl6)
	)
(:goal (and
(at plane3 city1)
(at person1 city0)
(at person2 city5)
(at person3 city4)
(at person4 city4)
))
)

