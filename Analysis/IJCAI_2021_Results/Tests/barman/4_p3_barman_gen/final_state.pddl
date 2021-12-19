(define (problem prob)
(:domain barman)
(:objects
shaker1 - shaker
left right - hand
shot1 shot2 shot3 shot4 shot5 - shot
ingredient1 ingredient2 ingredient3 - ingredient
cocktail1 cocktail2 cocktail3 - cocktail
dispenser1 dispenser2 dispenser3 - dispenser
l0 l1 l2 - level
)
	(:init
			(cocktail-part1 cocktail1 ingredient2)
			(cocktail-part1 cocktail2 ingredient2)
			(cocktail-part1 cocktail3 ingredient2)
			(cocktail-part2 cocktail1 ingredient1)
			(cocktail-part2 cocktail2 ingredient1)
			(cocktail-part2 cocktail3 ingredient3)
			(contains shot1 cocktail1)
			(contains shot3 cocktail3)
			(contains shot5 cocktail1)
			(dispenses dispenser1 ingredient1)
			(dispenses dispenser2 ingredient2)
			(dispenses dispenser3 ingredient3)
			(empty shaker1)
			(empty shot2)
			(empty shot4)
			(holding left shaker1)
			(holding right shot1)
			(next l0 l1)
			(next l1 l2)
			(ontable shot2)
			(ontable shot3)
			(ontable shot4)
			(ontable shot5)
			(shaker-empty-level shaker1 l0)
			(shaker-level shaker1 l0)
			(used shot2 ingredient3)
			(used shot4 ingredient1)
	)
(:goal
(and
(contains shot1 cocktail2)
(contains shot2 cocktail1)
(contains shot3 cocktail3)
(contains shot4 ingredient1)
)))









































































