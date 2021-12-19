

(define (problem BW-rand-10)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10  - block)
(:init
(handempty)
(on b1 b5)
(on b2 b8)
(on b3 b9)
(on b4 b10)
(on b5 b6)
(on b6 b2)
(on b7 b1)
(ontable b8)
(on b9 b7)
(ontable b10)
(clear b3)
(clear b4)
)
(:goal
(and
(on b2 b3)
(on b3 b8)
(on b4 b7)
(on b5 b2)
(on b6 b10)
(on b7 b1)
(on b8 b9)
(on b10 b5))
)
)


