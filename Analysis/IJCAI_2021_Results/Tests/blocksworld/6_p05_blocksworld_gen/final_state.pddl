(define (problem bw-rand-8)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8  - block)
	(:init
			(clear b1)
			(clear b4)
			(clear b6)
			(handempty)
			(on b2 b5)
			(on b3 b2)
			(on b4 b7)
			(on b6 b3)
			(on b7 b8)
			(ontable b1)
			(ontable b5)
			(ontable b8)
	)
(:goal
(and
(on b4 b1)
(on b5 b3)
(on b6 b7)
(on b7 b2)
(on b8 b6))
)
)

