(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 - table
d1 - disc
)
	(:init
			(clear d1)
			(clear peg1)
			(clear peg2)
			(clear peg3)
			(on d1 peg2)
			(smaller peg1 d1)
			(smaller peg2 d1)
			(smaller peg3 d1)
	)
(:goal
(and
(on d1 peg3)
)
)
)


