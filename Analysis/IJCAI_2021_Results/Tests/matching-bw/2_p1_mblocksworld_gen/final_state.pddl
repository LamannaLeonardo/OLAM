(define (problem matching-bw-typed-n5)
(:domain matching-bw-typed)
(:requirements :typing)
(:objects h1 h2 - hand b1 b2 b3 b4 b5  - block)
	(:init
			(block-negative b3)
			(block-negative b4)
			(block-negative b5)
			(block-positive b1)
			(block-positive b2)
			(clear b2)
			(clear b4)
			(clear b5)
			(hand-negative h2)
			(hand-positive h1)
			(holding h1 b3)
			(holding h2 b1)
			(on-table b2)
			(on-table b4)
			(on-table b5)
			(solid b3)
	)
(:goal
(and
(on b2 b5)
(on b3 b2)
(on b4 b3))
)
)





































