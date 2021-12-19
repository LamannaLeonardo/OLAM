(define (problem matching-bw-typed-n6)
(:domain matching-bw-typed)
(:requirements :typing)
(:objects h1 h2 - hand b1 b2 b3 b4 b5 b6  - block)
	(:init
			(block-negative b4)
			(block-negative b5)
			(block-negative b6)
			(block-positive b1)
			(block-positive b2)
			(block-positive b3)
			(clear b1)
			(clear b4)
			(empty h1)
			(empty h2)
			(hand-negative h2)
			(hand-positive h1)
			(on b1 b2)
			(on b2 b6)
			(on b4 b5)
			(on b5 b3)
			(on-table b3)
			(on-table b6)
			(solid b2)
			(solid b3)
			(solid b5)
			(solid b6)
	)
(:goal
(and
(on b1 b5)
(on b5 b2)
(on b6 b3))
)
)





















