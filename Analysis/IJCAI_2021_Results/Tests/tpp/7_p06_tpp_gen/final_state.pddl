(define (problem tpp)
(:domain tpp-propositional)
(:objects
goods1 goods2 goods3 goods4 goods5 goods6 - goods
truck1 truck2 truck3 truck4 - truck
market1 market2 market3 market4 - market
depot1 depot2 depot3 depot4 - depot
level0 level1 level2 level3 level4 - level)
	(:init
			(at truck1 market2)
			(at truck2 depot1)
			(at truck3 depot1)
			(at truck4 market3)
			(connected depot1 market4)
			(connected depot2 market3)
			(connected depot3 market1)
			(connected depot4 market2)
			(connected market1 depot3)
			(connected market1 market2)
			(connected market1 market3)
			(connected market1 market4)
			(connected market2 depot4)
			(connected market2 market1)
			(connected market3 depot2)
			(connected market3 market1)
			(connected market3 market4)
			(connected market4 depot1)
			(connected market4 market1)
			(connected market4 market3)
			(loaded goods1 truck1 level0)
			(loaded goods1 truck2 level0)
			(loaded goods1 truck3 level0)
			(loaded goods1 truck4 level0)
			(loaded goods2 truck1 level0)
			(loaded goods2 truck2 level0)
			(loaded goods2 truck3 level0)
			(loaded goods2 truck4 level0)
			(loaded goods3 truck1 level0)
			(loaded goods3 truck2 level0)
			(loaded goods3 truck3 level0)
			(loaded goods3 truck4 level0)
			(loaded goods4 truck1 level0)
			(loaded goods4 truck2 level0)
			(loaded goods4 truck3 level0)
			(loaded goods4 truck4 level0)
			(loaded goods5 truck1 level0)
			(loaded goods5 truck2 level0)
			(loaded goods5 truck3 level0)
			(loaded goods5 truck4 level0)
			(loaded goods6 truck1 level0)
			(loaded goods6 truck2 level0)
			(loaded goods6 truck3 level0)
			(loaded goods6 truck4 level0)
			(next level1 level0)
			(next level2 level1)
			(next level3 level2)
			(next level4 level3)
			(on-sale goods1 market1 level0)
			(on-sale goods1 market2 level0)
			(on-sale goods1 market3 level0)
			(on-sale goods1 market4 level0)
			(on-sale goods2 market1 level0)
			(on-sale goods2 market2 level0)
			(on-sale goods2 market3 level0)
			(on-sale goods2 market4 level1)
			(on-sale goods3 market1 level2)
			(on-sale goods3 market2 level0)
			(on-sale goods3 market3 level0)
			(on-sale goods3 market4 level1)
			(on-sale goods4 market1 level2)
			(on-sale goods4 market2 level0)
			(on-sale goods4 market3 level0)
			(on-sale goods4 market4 level1)
			(on-sale goods5 market1 level1)
			(on-sale goods5 market2 level0)
			(on-sale goods5 market3 level0)
			(on-sale goods5 market4 level0)
			(on-sale goods6 market1 level2)
			(on-sale goods6 market2 level0)
			(on-sale goods6 market3 level0)
			(on-sale goods6 market4 level0)
			(ready-to-load goods1 market1 level0)
			(ready-to-load goods1 market2 level1)
			(ready-to-load goods1 market3 level1)
			(ready-to-load goods1 market4 level0)
			(ready-to-load goods2 market1 level0)
			(ready-to-load goods2 market2 level1)
			(ready-to-load goods2 market3 level0)
			(ready-to-load goods2 market4 level0)
			(ready-to-load goods3 market1 level0)
			(ready-to-load goods3 market2 level1)
			(ready-to-load goods3 market3 level0)
			(ready-to-load goods3 market4 level0)
			(ready-to-load goods4 market1 level0)
			(ready-to-load goods4 market2 level1)
			(ready-to-load goods4 market3 level0)
			(ready-to-load goods4 market4 level0)
			(ready-to-load goods5 market1 level0)
			(ready-to-load goods5 market2 level1)
			(ready-to-load goods5 market3 level0)
			(ready-to-load goods5 market4 level0)
			(ready-to-load goods6 market1 level0)
			(ready-to-load goods6 market2 level2)
			(ready-to-load goods6 market3 level0)
			(ready-to-load goods6 market4 level0)
			(stored goods1 level1)
			(stored goods2 level0)
			(stored goods3 level0)
			(stored goods4 level0)
			(stored goods5 level2)
			(stored goods6 level0)
	)
(:goal (and
(stored goods1 level3)
(stored goods2 level1)
(stored goods3 level1)
(stored goods4 level2)
(stored goods5 level2)
(stored goods6 level2)))
)
























