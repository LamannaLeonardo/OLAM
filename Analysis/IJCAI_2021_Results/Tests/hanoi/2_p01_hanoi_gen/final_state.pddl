(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 - table
d1 d2 - disc
)
	(:init
			(clear d1)
			(clear d2)
			(clear peg2)
			(clear peg3)
			(on d1 d1)
			(on d2 peg1)
			(smaller d1 d1)
			(smaller d2 d1)
			(smaller peg1 d1)
			(smaller peg1 d2)
			(smaller peg2 d1)
			(smaller peg2 d2)
			(smaller peg3 d1)
			(smaller peg3 d2)
	)
(:goal
(and
(on d2 peg3)
(on d1 d2)
)
)
)



