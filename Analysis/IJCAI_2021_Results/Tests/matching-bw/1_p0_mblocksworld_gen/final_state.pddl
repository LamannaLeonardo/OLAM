(define (problem matching-bw-typed-n4)
(:domain matching-bw-typed)
(:requirements :typing)
(:objects h1 h2 - hand b1 b2 b3 b4  - block)
	(:init
			(block-negative b3)
			(block-negative b4)
			(block-positive b1)
			(block-positive b2)
			(clear b1)
			(clear b2)
			(clear b3)
			(clear b4)
			(empty h1)
			(empty h2)
			(hand-negative h2)
			(hand-positive h1)
			(on-table b1)
			(on-table b2)
			(on-table b3)
			(on-table b4)
	)
(:goal
(and
(on b1 b3)
(on b2 b1)
(on b3 b4))
)
)









































































