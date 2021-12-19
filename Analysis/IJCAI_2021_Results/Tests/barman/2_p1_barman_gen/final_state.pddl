(define (problem prob)
(:domain barman)
(:objects
shaker1 - shaker
left right - hand
shot1 shot2 shot3 - shot
ingredient1 ingredient2 ingredient3 - ingredient
cocktail1 cocktail2 - cocktail
dispenser1 dispenser2 dispenser3 - dispenser
l0 l1 l2 - level
)
	(:init
			(clean shot2)
			(clean shot3)
			(cocktail-part1 cocktail1 ingredient3)
			(cocktail-part1 cocktail2 ingredient3)
			(cocktail-part2 cocktail1 ingredient2)
			(cocktail-part2 cocktail2 ingredient1)
			(contains shaker1 ingredient3)
			(dispenses dispenser1 ingredient1)
			(dispenses dispenser2 ingredient2)
			(dispenses dispenser3 ingredient3)
			(empty shot1)
			(empty shot2)
			(empty shot3)
			(holding left shaker1)
			(holding right shot1)
			(next l0 l1)
			(next l1 l2)
			(ontable shot2)
			(ontable shot3)
			(shaker-empty-level shaker1 l0)
			(shaker-level shaker1 l2)
			(unshaked shaker1)
			(used shot1 ingredient3)
	)
(:goal
(and
(contains shot1 cocktail1)
(contains shot2 cocktail2)
)))









