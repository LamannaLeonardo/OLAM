


(define (problem grid-x4-y5-t2-k22-l33-p100)
(:domain grid)
(:objects 
        f0-0f f1-0f f2-0f f3-0f 
        f0-1f f1-1f f2-1f f3-1f 
        f0-2f f1-2f f2-2f f3-2f 
        f0-3f f1-3f f2-3f f3-3f 
        f0-4f f1-4f f2-4f f3-4f - place
        shape0 shape1 - shape
        key0-0 key0-1 
        key1-0 key1-1 - key
)
(:init
(arm-empty)
(key-shape key0-0 shape0)
(key-shape key0-1 shape0)
(key-shape key1-0 shape1)
(key-shape key1-1 shape1)
(conn f0-0f f1-0f)
(conn f1-0f f2-0f)
(conn f2-0f f3-0f)
(conn f0-1f f1-1f)
(conn f1-1f f2-1f)
(conn f2-1f f3-1f)
(conn f0-2f f1-2f)
(conn f1-2f f2-2f)
(conn f2-2f f3-2f)
(conn f0-3f f1-3f)
(conn f1-3f f2-3f)
(conn f2-3f f3-3f)
(conn f0-4f f1-4f)
(conn f1-4f f2-4f)
(conn f2-4f f3-4f)
(conn f0-0f f0-1f)
(conn f1-0f f1-1f)
(conn f2-0f f2-1f)
(conn f3-0f f3-1f)
(conn f0-1f f0-2f)
(conn f1-1f f1-2f)
(conn f2-1f f2-2f)
(conn f3-1f f3-2f)
(conn f0-2f f0-3f)
(conn f1-2f f1-3f)
(conn f2-2f f2-3f)
(conn f3-2f f3-3f)
(conn f0-3f f0-4f)
(conn f1-3f f1-4f)
(conn f2-3f f2-4f)
(conn f3-3f f3-4f)
(conn f1-0f f0-0f)
(conn f2-0f f1-0f)
(conn f3-0f f2-0f)
(conn f1-1f f0-1f)
(conn f2-1f f1-1f)
(conn f3-1f f2-1f)
(conn f1-2f f0-2f)
(conn f2-2f f1-2f)
(conn f3-2f f2-2f)
(conn f1-3f f0-3f)
(conn f2-3f f1-3f)
(conn f3-3f f2-3f)
(conn f1-4f f0-4f)
(conn f2-4f f1-4f)
(conn f3-4f f2-4f)
(conn f0-1f f0-0f)
(conn f1-1f f1-0f)
(conn f2-1f f2-0f)
(conn f3-1f f3-0f)
(conn f0-2f f0-1f)
(conn f1-2f f1-1f)
(conn f2-2f f2-1f)
(conn f3-2f f3-1f)
(conn f0-3f f0-2f)
(conn f1-3f f1-2f)
(conn f2-3f f2-2f)
(conn f3-3f f3-2f)
(conn f0-4f f0-3f)
(conn f1-4f f1-3f)
(conn f2-4f f2-3f)
(conn f3-4f f3-3f)
(open f0-0f)
(open f1-0f)
(open f2-0f)
(open f3-0f)
(open f0-1f)
(open f1-1f)
(open f2-1f)
(open f3-1f)
(open f0-2f)
(open f2-2f)
(open f2-3f)
(open f3-3f)
(open f1-4f)
(open f2-4f)
(locked f0-4f)
(lock-shape f0-4f shape0)
(locked f3-4f)
(lock-shape f3-4f shape0)
(locked f3-2f)
(lock-shape f3-2f shape0)
(locked f1-3f)
(lock-shape f1-3f shape1)
(locked f0-3f)
(lock-shape f0-3f shape1)
(locked f1-2f)
(lock-shape f1-2f shape1)
(at key0-0 f3-3f)
(at key0-1 f3-1f)
(at key1-0 f1-1f)
(at key1-1 f2-3f)
(at-robot f3-0f)
)
(:goal
(and
(at key0-0 f0-4f)
(at key0-1 f1-2f)
(at key1-0 f0-3f)
(at key1-1 f3-3f)
)
)
)


