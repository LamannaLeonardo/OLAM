(define (problem bw-rand-7)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7  - block)
	(:init
			(clear b1)
			(clear b3)
			(handempty)
			(on b1 b7)
			(on b2 b5)
			(on b4 b6)
			(on b5 b4)
			(on b7 b2)
			(ontable b3)
			(ontable b6)
	)
(:goal
(and
(on b3 b2)
(on b6 b4)
(on b7 b6))
)
)

