(define (domain hanoi)
 (:requirements :strips :typing)
 (:types disc table - platform)
 (:predicates (clear ?x - platform)
    (on ?x - disc ?y - platform)
    (smaller ?x - platform ?y - disc))

 (:action move
    :parameters (?disc - disc ?from - platform ?to - platform)
    :precondition (and (smaller ?to ?disc)
    (on ?disc ?from)
    (clear ?disc)
    (clear ?to))
    :effect (and (clear ?from)
    (on ?disc ?to)
    (not (on ?disc ?from)))
    )

    )
