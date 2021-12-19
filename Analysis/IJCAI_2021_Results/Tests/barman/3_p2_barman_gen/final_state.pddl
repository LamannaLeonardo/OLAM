(define (problem prob)
(:domain barman)
(:objects
shaker1 - shaker
left right - hand
shot1 shot2 shot3 shot4 - shot
ingredient1 ingredient2 ingredient3 - ingredient
cocktail1 cocktail2 cocktail3 - cocktail
dispenser1 dispenser2 dispenser3 - dispenser
l0 l1 l2 - level
)
	(:init
			(cocktail-part1 cocktail1 ingredient1)
			(cocktail-part1 cocktail2 ingredient1)
			(cocktail-part1 cocktail3 ingredient3)
			(cocktail-part2 cocktail1 ingredient2)
			(cocktail-part2 cocktail2 ingredient3)
			(cocktail-part2 cocktail3 ingredient2)
			(contains shaker1 ingredient1)
			(contains shot4 cocktail3)
			(dispenses dispenser1 ingredient1)
			(dispenses dispenser2 ingredient2)
			(dispenses dispenser3 ingredient3)
			(empty shot1)
			(empty shot2)
			(empty shot3)
			(holding left shot1)
			(holding right shaker1)
			(next l0 l1)
			(next l1 l2)
			(ontable shot2)
			(ontable shot3)
			(ontable shot4)
			(shaker-empty-level shaker1 l0)
			(shaker-level shaker1 l2)
			(unshaked shaker1)
			(used shot3 ingredient1)
	)
(:goal
(and
(contains shot1 cocktail2)
(contains shot2 cocktail1)
(contains shot3 cocktail3)
)))






























































