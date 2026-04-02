


(define (problem grid-x3-y4-t1-k1-l2-p100)
(:domain grid)
(:objects 
        f0-0f f1-0f f2-0f 
        f0-1f f1-1f f2-1f 
        f0-2f f1-2f f2-2f 
        f0-3f f1-3f f2-3f - place
        shape0 - shape
        key0-0 - key
)
(:init
(arm-empty)
(key-shape key0-0 shape0)
(conn f0-0f f1-0f)
(conn f1-0f f2-0f)
(conn f0-1f f1-1f)
(conn f1-1f f2-1f)
(conn f0-2f f1-2f)
(conn f1-2f f2-2f)
(conn f0-3f f1-3f)
(conn f1-3f f2-3f)
(conn f0-0f f0-1f)
(conn f1-0f f1-1f)
(conn f2-0f f2-1f)
(conn f0-1f f0-2f)
(conn f1-1f f1-2f)
(conn f2-1f f2-2f)
(conn f0-2f f0-3f)
(conn f1-2f f1-3f)
(conn f2-2f f2-3f)
(conn f1-0f f0-0f)
(conn f2-0f f1-0f)
(conn f1-1f f0-1f)
(conn f2-1f f1-1f)
(conn f1-2f f0-2f)
(conn f2-2f f1-2f)
(conn f1-3f f0-3f)
(conn f2-3f f1-3f)
(conn f0-1f f0-0f)
(conn f1-1f f1-0f)
(conn f2-1f f2-0f)
(conn f0-2f f0-1f)
(conn f1-2f f1-1f)
(conn f2-2f f2-1f)
(conn f0-3f f0-2f)
(conn f1-3f f1-2f)
(conn f2-3f f2-2f)
(open f0-0f)
(open f1-0f)
(open f2-0f)
(open f0-1f)
(open f0-2f)
(open f1-2f)
(open f2-2f)
(open f0-3f)
(open f1-3f)
(open f2-3f)
(locked f2-1f)
(lock-shape f2-1f shape0)
(locked f1-1f)
(lock-shape f1-1f shape0)
(at key0-0 f1-0f)
(at-robot f0-2f)
)
(:goal
(and
(at key0-0 f2-3f)
)
)
)


