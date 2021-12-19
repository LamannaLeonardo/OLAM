(define (problem tpp)
(:domain tpp-propositional)
(:objects
goods1 goods2 goods3 - goods
truck1 - truck
market1 - market
depot1 - depot
level0 level1 - level)
	(:init
			(at truck1 depot1)
			(connected depot1 market1)
			(connected market1 depot1)
			(loaded goods1 truck1 level0)
			(loaded goods2 truck1 level0)
			(loaded goods3 truck1 level0)
			(next level1 level0)
			(on-sale goods1 market1 level0)
			(on-sale goods2 market1 level0)
			(on-sale goods3 market1 level1)
			(ready-to-load goods1 market1 level0)
			(ready-to-load goods2 market1 level0)
			(ready-to-load goods3 market1 level0)
			(stored goods1 level1)
			(stored goods2 level1)
			(stored goods3 level0)
	)
(:goal (and
(stored goods1 level1)
(stored goods2 level1)
(stored goods3 level1)))
)









































