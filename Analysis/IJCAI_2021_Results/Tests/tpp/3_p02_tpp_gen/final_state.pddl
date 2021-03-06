(define (problem tpp)
(:domain tpp-propositional)
(:objects
goods1 goods2 goods3 goods4 - goods
truck1 truck2 - truck
market1 market2 - market
depot1 depot2 - depot
level0 level1 level2 - level)
	(:init
			(at truck1 depot1)
			(at truck2 depot2)
			(connected depot1 market2)
			(connected depot2 market1)
			(connected market1 depot2)
			(connected market1 market2)
			(connected market2 depot1)
			(connected market2 market1)
			(loaded goods1 truck1 level0)
			(loaded goods1 truck2 level0)
			(loaded goods2 truck1 level0)
			(loaded goods2 truck2 level0)
			(loaded goods3 truck1 level0)
			(loaded goods3 truck2 level0)
			(loaded goods4 truck1 level0)
			(loaded goods4 truck2 level0)
			(next level1 level0)
			(next level2 level1)
			(on-sale goods1 market1 level0)
			(on-sale goods1 market2 level0)
			(on-sale goods2 market1 level0)
			(on-sale goods2 market2 level0)
			(on-sale goods3 market1 level0)
			(on-sale goods3 market2 level0)
			(on-sale goods4 market1 level0)
			(on-sale goods4 market2 level0)
			(ready-to-load goods1 market1 level0)
			(ready-to-load goods1 market2 level0)
			(ready-to-load goods2 market1 level0)
			(ready-to-load goods2 market2 level0)
			(ready-to-load goods3 market1 level0)
			(ready-to-load goods3 market2 level0)
			(ready-to-load goods4 market1 level0)
			(ready-to-load goods4 market2 level0)
			(stored goods1 level2)
			(stored goods2 level2)
			(stored goods3 level2)
			(stored goods4 level2)
	)
(:goal (and
(stored goods1 level1)
(stored goods2 level2)
(stored goods3 level2)
(stored goods4 level1)))
)











































