(define (problem depotprob134536825) (:domain depots)
(:objects
depot0 depot1 - depot
distributor0 distributor1 - distributor
truck0 - truck
pallet0 pallet1 pallet2 pallet3 - pallet
crate0 crate1 crate2 - crate
hoist0 hoist1 hoist2 hoist3 - hoist)
	(:init
			(at crate0 depot0)
			(at crate1 distributor1)
			(at crate2 depot0)
			(at hoist0 depot0)
			(at hoist1 depot1)
			(at hoist2 distributor0)
			(at hoist3 distributor1)
			(at pallet0 depot0)
			(at pallet1 depot1)
			(at pallet2 distributor0)
			(at pallet3 distributor1)
			(at truck0 distributor0)
			(available hoist0)
			(available hoist1)
			(available hoist2)
			(available hoist3)
			(clear crate1)
			(clear crate2)
			(clear pallet1)
			(clear pallet2)
			(on crate0 pallet0)
			(on crate1 pallet3)
			(on crate2 crate0)
	)
(:goal (and
(on crate1 pallet1)
(on crate2 crate1)
)
))

