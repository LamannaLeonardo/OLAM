(define (problem prob)
(:domain barman)
(:objects
shaker1 - shaker
left right - hand
shot1 shot2 - shot
ingredient1 ingredient2 - ingredient
cocktail1 - cocktail
dispenser1 dispenser2 - dispenser
l0 l1 l2 - level
)
	(:init
			(clean shot2)
			(cocktail-part1 cocktail1 ingredient2)
			(cocktail-part2 cocktail1 ingredient1)
			(contains shaker1 ingredient2)
			(contains shot1 ingredient2)
			(dispenses dispenser1 ingredient1)
			(dispenses dispenser2 ingredient2)
			(empty shot2)
			(handempty left)
			(handempty right)
			(next l0 l1)
			(next l1 l2)
			(ontable shaker1)
			(ontable shot1)
			(ontable shot2)
			(shaker-empty-level shaker1 l0)
			(shaker-level shaker1 l2)
			(unshaked shaker1)
			(used shot1 ingredient2)
	)
(:goal
(and
(contains shot1 cocktail1)
)))
















































































































