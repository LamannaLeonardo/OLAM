(define (problem ZTRAVEL-6-5)
(:domain zeno-travel)
(:objects
	plane1 - aircraft
	plane2 - aircraft
	plane3 - aircraft
	plane4 - aircraft
	plane5 - aircraft
	plane6 - aircraft
	person1 - person
	person2 - person
	person3 - person
	person4 - person
	person5 - person
	city0 - city
	city1 - city
	city2 - city
	city3 - city
	city4 - city
	city5 - city
	city6 - city
	city7 - city
	fl0 - flevel
	fl1 - flevel
	fl2 - flevel
	fl3 - flevel
	fl4 - flevel
	fl5 - flevel
	fl6 - flevel
	)
(:init
	(at plane1 city7)
	(fuel-level plane1 fl5)
	(at plane2 city6)
	(fuel-level plane2 fl3)
	(at plane3 city4)
	(fuel-level plane3 fl2)
	(at plane4 city7)
	(fuel-level plane4 fl1)
	(at plane5 city3)
	(fuel-level plane5 fl4)
	(at plane6 city7)
	(fuel-level plane6 fl5)
	(at person1 city3)
	(at person2 city6)
	(at person3 city3)
	(at person4 city7)
	(at person5 city5)
	(next fl0 fl1)
	(next fl1 fl2)
	(next fl2 fl3)
	(next fl3 fl4)
	(next fl4 fl5)
	(next fl5 fl6)
)
(:goal (and
	(at plane2 city4)
	(at plane6 city6)
	(at person1 city7)
	(at person2 city5)
	(at person3 city1)
	(at person4 city1)
	(at person5 city3)
	))

)
