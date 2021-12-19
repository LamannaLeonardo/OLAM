(define (problem bw-rand-4)
(:domain blocksworld)
(:objects b1 b2 b3 b4  - block)
	(:init
			(clear b1)
			(clear b4)
			(handempty)
			(on b1 b3)
			(on b4 b2)
			(ontable b2)
			(ontable b3)
	)
(:goal
(and
(on b2 b4)
(on b3 b2))
)
)

