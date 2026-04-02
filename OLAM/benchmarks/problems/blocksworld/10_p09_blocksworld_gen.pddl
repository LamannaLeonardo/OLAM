

(define (problem BW-rand-12)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12  - block)
(:init
(handempty)
(ontable b1)
(on b2 b1)
(on b3 b9)
(ontable b4)
(on b5 b10)
(on b6 b7)
(on b7 b2)
(on b8 b4)
(on b9 b12)
(on b10 b11)
(on b11 b8)
(on b12 b6)
(clear b3)
(clear b5)
)
(:goal
(and
(on b1 b4)
(on b2 b12)
(on b3 b5)
(on b4 b7)
(on b5 b1)
(on b6 b2)
(on b7 b6)
(on b8 b11)
(on b11 b9)
(on b12 b8))
)
)


