


(define (problem grid-x4-y3-t2-k11-l22-p100)
(:domain grid)
(:objects 
        f0-0f f1-0f f2-0f f3-0f 
        f0-1f f1-1f f2-1f f3-1f 
        f0-2f f1-2f f2-2f f3-2f - place
        shape0 shape1 - shape
        key0-0 
        key1-0 - key
)
(:init
(arm-empty)
(key-shape key0-0 shape0)
(key-shape key1-0 shape1)
(conn f0-0f f1-0f)
(conn f1-0f f2-0f)
(conn f2-0f f3-0f)
(conn f0-1f f1-1f)
(conn f1-1f f2-1f)
(conn f2-1f f3-1f)
(conn f0-2f f1-2f)
(conn f1-2f f2-2f)
(conn f2-2f f3-2f)
(conn f0-0f f0-1f)
(conn f1-0f f1-1f)
(conn f2-0f f2-1f)
(conn f3-0f f3-1f)
(conn f0-1f f0-2f)
(conn f1-1f f1-2f)
(conn f2-1f f2-2f)
(conn f3-1f f3-2f)
(conn f1-0f f0-0f)
(conn f2-0f f1-0f)
(conn f3-0f f2-0f)
(conn f1-1f f0-1f)
(conn f2-1f f1-1f)
(conn f3-1f f2-1f)
(conn f1-2f f0-2f)
(conn f2-2f f1-2f)
(conn f3-2f f2-2f)
(conn f0-1f f0-0f)
(conn f1-1f f1-0f)
(conn f2-1f f2-0f)
(conn f3-1f f3-0f)
(conn f0-2f f0-1f)
(conn f1-2f f1-1f)
(conn f2-2f f2-1f)
(conn f3-2f f3-1f)
(open f1-0f)
(open f0-1f)
(open f1-1f)
(open f2-1f)
(open f3-1f)
(open f0-2f)
(open f2-2f)
(open f3-2f)
(locked f1-2f)
(lock-shape f1-2f shape0)
(locked f2-0f)
(lock-shape f2-0f shape0)
(locked f3-0f)
(lock-shape f3-0f shape1)
(locked f0-0f)
(lock-shape f0-0f shape1)
(at key0-0 f1-0f)
(at key1-0 f1-1f)
(at-robot f0-1f)
)
(:goal
(and
(at key0-0 f2-2f)
(at key1-0 f0-2f)
)
)
)


