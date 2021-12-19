(define (problem bw-rand-3)
(:domain blocksworld)
(:objects b1 b2 b3  - block)
	(:init
			(clear b3)
			(handempty)
			(on b1 b2)
			(on b3 b1)
			(ontable b2)
	)
(:goal
(and
(on b3 b2))
)
)











