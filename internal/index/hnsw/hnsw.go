package hnsw

// DistFn computes the int32 squared distance between the (translated) query
// and the member at local id `mid`. Implementations close over the per-query
// state (qRes) and the index's member residual buffer.
type DistFn func(mid uint16) int32

// Graph is one HNSW graph (one cluster). All edge lists use cluster-local ids.
//
// Edges layout:
//
//	Level 0: every node n has Edges[0][n*M : n*M+Degree[0][n]] neighbors.
//	Level L≥1: only NodeIds[L] are present; for the i-th node in NodeIds[L],
//	           neighbors at Edges[L][i*M : i*M+Degree[L][i]].
//
// The Search routine doesn't need NodeIds for level 0 (slot = id). For L≥1
// it does binary search on NodeIds[L].
type Graph struct {
	N         uint16     // number of nodes (cluster size)
	M         uint8      // max neighbors per layer
	MaxLevel  uint8      // highest level used
	Entry     uint16     // entry point at top layer
	NodeIds   [][]uint16 // [level][i] = node id at index i; nil/empty for level 0
	Edges     [][]uint16 // [level] flat: len = nodesAtLevel * M; pad value 0xFFFF
	Degree    [][]uint8  // [level][i] = number of valid edges at that slot
}

// Search finds the K nearest neighbors of the (translated) query using
// the supplied DistFn. Writes into `out` (pre-allocated MaxHeap, reset by
// the caller). Visited set is a bitmap sized to g.N; callers pass scratch.
//
// `ef` is the beam width at layer 0 (≥ K).
//
// Algorithm: standard HNSW. At each level above 0 do greedy descent (always
// move to the strictly-closer neighbor); at level 0 do an ef-sized beam
// search collecting up to ef candidates, then truncate to K.
func (g *Graph) Search(
	dist DistFn,
	K, ef int,
	visited []uint64,
	cand *MinHeap,
	out *MaxHeap,
) {
	clearBitmap(visited, int(g.N))
	out.Reset()
	cand.Reset()

	cur := g.Entry
	curD := dist(cur)

	// Greedy descent from top level down to level 1.
	for L := int(g.MaxLevel); L >= 1; L-- {
		improved := true
		for improved {
			improved = false
			for _, nb := range g.neighbors(uint8(L), cur) {
				if nb == 0xFFFF {
					break
				}
				d := dist(nb)
				if d < curD {
					curD = d
					cur = nb
					improved = true
				}
			}
		}
	}

	// Beam search at level 0 starting from `cur`.
	setBit(visited, int(cur))
	cand.Push(Entry{Dist: curD, ID: cur})
	out.Push(Entry{Dist: curD, ID: cur})

	for cand.Len() > 0 {
		c := cand.Pop()
		// If the closest unexpanded is worse than the worst in `out` AND `out`
		// is at full ef, no further improvement is possible.
		if out.Len() >= ef && c.Dist > out.Top().Dist {
			break
		}
		for _, nb := range g.neighbors(0, c.ID) {
			if nb == 0xFFFF {
				break
			}
			if testBit(visited, int(nb)) {
				continue
			}
			setBit(visited, int(nb))
			d := dist(nb)
			if out.Len() < ef || d < out.Top().Dist {
				cand.Push(Entry{Dist: d, ID: nb})
				out.Push(Entry{Dist: d, ID: nb})
				for out.Len() > ef {
					out.Pop()
				}
			}
		}
	}

	// Truncate from `ef` down to `K`.
	for out.Len() > K {
		out.Pop()
	}
}

// neighbors returns the edge slice for `id` at level `L`. For level 0 this
// is a direct index; for L≥1 it's a binary search on NodeIds[L].
func (g *Graph) neighbors(L uint8, id uint16) []uint16 {
	if L == 0 {
		base := int(id) * int(g.M)
		n := int(g.Degree[0][id])
		return g.Edges[0][base : base+n]
	}
	// Binary search NodeIds[L] for id.
	ids := g.NodeIds[L]
	lo, hi := 0, len(ids)-1
	for lo <= hi {
		mid := (lo + hi) / 2
		if ids[mid] == id {
			base := mid * int(g.M)
			n := int(g.Degree[L][mid])
			return g.Edges[L][base : base+n]
		}
		if ids[mid] < id {
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}
	return nil
}

func clearBitmap(b []uint64, n int) {
	words := (n + 63) >> 6
	for i := 0; i < words; i++ {
		b[i] = 0
	}
}
func setBit(b []uint64, i int)        { b[i>>6] |= 1 << uint(i&63) }
func testBit(b []uint64, i int) bool  { return b[i>>6]&(1<<uint(i&63)) != 0 }
