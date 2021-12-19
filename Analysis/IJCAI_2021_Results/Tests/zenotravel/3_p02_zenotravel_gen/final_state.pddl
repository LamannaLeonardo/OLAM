(define (problem ztravel-3-2)
(:domain zeno-travel)
(:objects
plane1 - aircraft
plane2 - aircraft
plane3 - aircraft
person1 - person
person2 - person
city0 - city
city1 - city
city2 - city
fl0 - flevel
fl1 - flevel
fl2 - flevel
fl3 - flevel
fl4 - flevel
fl5 - flevel
fl6 - flevel
)
	(:init
			(at person1 city1)
			(at person2 city2)
			(at plane1 city2)
			(at plane2 city2)
			(at plane3 city1)
			(fuel-level plane1 fl1)
			(fuel-level plane2 fl1)
			(fuel-level plane3 fl0)
			(next fl0 fl1)
			(next fl1 fl2)
			(next fl2 fl3)
			(next fl3 fl4)
			(next fl4 fl5)
			(next fl5 fl6)
	)
(:goal (and
(at person1 city2)
(at person2 city1)
))
)

