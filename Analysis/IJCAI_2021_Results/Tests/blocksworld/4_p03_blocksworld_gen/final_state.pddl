(define (problem bw-rand-6)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6  - block)
	(:init
			(clear b2)
			(clear b3)
			(clear b4)
			(handempty)
			(on b3 b5)
			(on b4 b1)
			(on b5 b6)
			(ontable b1)
			(ontable b2)
			(ontable b6)
	)
(:goal
(and
(on b1 b2)
(on b2 b6)
(on b3 b4))
)
)

