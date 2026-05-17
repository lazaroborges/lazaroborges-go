package hnsw

import (
	"math"
	"math/rand"
)

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
func setBit(b []uint64, i int)       { b[i>>6] |= 1 << uint(i&63) }
func testBit(b []uint64, i int) bool { return b[i>>6]&(1<<uint(i&63)) != 0 }

// BuildDist computes int32 squared distance between two member local ids,
// used only during graph construction.
type BuildDist func(a, b uint16) int32

// Build constructs an HNSW graph over N nodes (ids 0..N-1) using `dist`
// for all comparisons. M is the per-layer neighbor cap. efConstruction is
// the build-time beam width (larger = better recall, slower build).
//
// Layer assignment: random level = floor(-ln(uniform()) * mL) where
// mL = 1/ln(M). Capped at MaxLevelCap = 7.
//
// Edge selection at insertion: select-neighbors-simple from Malkov 2018
// (keep the M closest candidates after reverse-edge updates).
func Build(N uint16, M uint8, efConstruction int, seed int64, dist BuildDist) *Graph {
	const MaxLevelCap = 7
	mL := 1.0 / math.Log(float64(M))
	rng := rand.New(rand.NewSource(seed))

	// Assign levels.
	levels := make([]uint8, N)
	maxLevel := uint8(0)
	for i := uint16(0); i < N; i++ {
		u := rng.Float64()
		if u == 0 {
			u = 1e-12
		}
		L := int(math.Floor(-math.Log(u) * mL))
		if L > MaxLevelCap {
			L = MaxLevelCap
		}
		levels[i] = uint8(L)
		if uint8(L) > maxLevel {
			maxLevel = uint8(L)
		}
	}

	// Allocate graph. NodeIds[L] will be filled in order of insertion.
	g := &Graph{
		N:        N,
		M:        M,
		MaxLevel: maxLevel,
		Entry:    0,
		NodeIds:  make([][]uint16, maxLevel+1),
		Edges:    make([][]uint16, maxLevel+1),
		Degree:   make([][]uint8, maxLevel+1),
	}

	// Level 0: all nodes present; allocate full flat arrays up front.
	g.Edges[0] = make([]uint16, int(N)*int(M))
	g.Degree[0] = make([]uint8, N)
	for i := range g.Edges[0] {
		g.Edges[0][i] = 0xFFFF
	}

	// Levels 1..maxLevel: count how many nodes will be present (level i ≥ L).
	for L := uint8(1); L <= maxLevel; L++ {
		var nodes []uint16
		for i := uint16(0); i < N; i++ {
			if levels[i] >= L {
				nodes = append(nodes, i)
			}
		}
		g.NodeIds[L] = nodes
		g.Edges[L] = make([]uint16, len(nodes)*int(M))
		g.Degree[L] = make([]uint8, len(nodes))
		for i := range g.Edges[L] {
			g.Edges[L][i] = 0xFFFF
		}
	}

	// Insert nodes one by one. Maintain the current entry point and its level.
	entryLevel := levels[0]

	visited := make([]uint64, (int(N)+63)>>6)
	cand := MinHeap{}
	work := MaxHeap{}

	for newID := uint16(1); newID < N; newID++ {
		newLevel := levels[newID]

		// Greedy descent from entry level down to newLevel+1 (one best-neighbor
		// step per level — just moving the cursor, no ef-expansion needed).
		cur := g.Entry
		curD := dist(cur, newID)
		for L := int(entryLevel); L > int(newLevel); L-- {
			improved := true
			for improved {
				improved = false
				for _, nb := range g.neighbors(uint8(L), cur) {
					if nb == 0xFFFF {
						break
					}
					d := dist(nb, newID)
					if d < curD {
						curD = d
						cur = nb
						improved = true
					}
				}
			}
		}

		// For each level from min(newLevel, entryLevel) down to 0: ef-search
		// then connect.
		startL := newLevel
		if entryLevel < startL {
			startL = entryLevel
		}
		for L := int(startL); L >= 0; L-- {
			// Clear only the bits used so far (up to newID, not N).
			clearBitmapN(visited, int(newID))
			cand.Reset()
			work.Reset()

			setBit(visited, int(cur))
			cand.Push(Entry{Dist: curD, ID: cur})
			work.Push(Entry{Dist: curD, ID: cur})

			for cand.Len() > 0 {
				c := cand.Pop()
				if work.Len() >= efConstruction && c.Dist > work.Top().Dist {
					break
				}
				for _, nb := range g.neighbors(uint8(L), c.ID) {
					if nb == 0xFFFF {
						break
					}
					if testBit(visited, int(nb)) {
						continue
					}
					setBit(visited, int(nb))
					d := dist(nb, newID)
					if work.Len() < efConstruction || d < work.Top().Dist {
						cand.Push(Entry{Dist: d, ID: nb})
						work.Push(Entry{Dist: d, ID: nb})
						for work.Len() > efConstruction {
							work.Pop()
						}
					}
				}
			}

			// Select up to M neighbors (simple heuristic: keep M closest).
			items := append([]Entry(nil), work.Items()...)
			sortAscending(items)
			limit := int(M)
			if len(items) < limit {
				limit = len(items)
			}
			chosen := items[:limit]

			// Connect newID → chosen.
			setEdges(g, L, newID, chosen)
			// Reverse edges: chosen → newID, with pruning.
			for _, c := range chosen {
				addEdgeWithPrune(g, L, c.ID, newID, dist)
			}

			// Update cur for the next-lower level.
			if len(items) > 0 {
				cur = items[0].ID
				curD = items[0].Dist
			}
		}

		// Promote entry if newID reaches a higher level.
		if newLevel > entryLevel {
			g.Entry = newID
			entryLevel = newLevel
		}
	}

	return g
}

// clearBitmapN zeroes the bitmap for indices 0..n-1.
func clearBitmapN(b []uint64, n int) {
	words := (n + 63) >> 6
	for i := 0; i < words; i++ {
		b[i] = 0
	}
}

func sortAscending(items []Entry) {
	for i := 1; i < len(items); i++ {
		for j := i; j > 0 && items[j].Dist < items[j-1].Dist; j-- {
			items[j], items[j-1] = items[j-1], items[j]
		}
	}
}

// setEdges writes the neighbor list for `node` at level L.
// For L=0 the slot index equals node; for L≥1 it's looked up via NodeIds.
func setEdges(g *Graph, L int, node uint16, nbs []Entry) {
	if L == 0 {
		base := int(node) * int(g.M)
		for i := 0; i < int(g.M); i++ {
			g.Edges[0][base+i] = 0xFFFF
		}
		for i, e := range nbs {
			g.Edges[0][base+i] = e.ID
		}
		g.Degree[0][node] = uint8(len(nbs))
		return
	}
	slot := buildBinSearch(g.NodeIds[L], node)
	base := slot * int(g.M)
	for i := 0; i < int(g.M); i++ {
		g.Edges[L][base+i] = 0xFFFF
	}
	for i, e := range nbs {
		g.Edges[L][base+i] = e.ID
	}
	g.Degree[L][slot] = uint8(len(nbs))
}

// addEdgeWithPrune adds `target` to `node`'s neighbor list at level L.
// If the list is full, recompute the M closest among (current + target)
// and keep those.
func addEdgeWithPrune(g *Graph, L int, node, target uint16, dist BuildDist) {
	var (
		base   int
		deg    uint8
		edges  []uint16
		degArr []uint8
		slot   int
	)
	if L == 0 {
		slot = int(node)
		base = slot * int(g.M)
		deg = g.Degree[0][node]
		edges = g.Edges[0]
		degArr = g.Degree[0]
	} else {
		slot = buildBinSearch(g.NodeIds[L], node)
		base = slot * int(g.M)
		deg = g.Degree[L][slot]
		edges = g.Edges[L]
		degArr = g.Degree[L]
	}

	if int(deg) < int(g.M) {
		edges[base+int(deg)] = target
		degArr[slot] = deg + 1
		return
	}
	// Full: keep M closest among existing + target.
	cands := make([]Entry, 0, int(g.M)+1)
	for i := 0; i < int(g.M); i++ {
		cands = append(cands, Entry{Dist: dist(node, edges[base+i]), ID: edges[base+i]})
	}
	cands = append(cands, Entry{Dist: dist(node, target), ID: target})
	sortAscending(cands)
	for i := 0; i < int(g.M); i++ {
		edges[base+i] = cands[i].ID
	}
	// Degree stays at M.
}

// buildBinSearch finds the index of x in a sorted []uint16 (NodeIds[L]).
// Returns -1 if not found (should never happen during a correct build).
func buildBinSearch(ids []uint16, x uint16) int {
	lo, hi := 0, len(ids)-1
	for lo <= hi {
		m := (lo + hi) / 2
		if ids[m] == x {
			return m
		}
		if ids[m] < x {
			lo = m + 1
		} else {
			hi = m - 1
		}
	}
	return -1
}
