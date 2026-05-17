// Package hnsw implements a per-cluster Hierarchical Navigable Small World
// graph for the IVF-HNSW hybrid index. All distances are int32; entries
// carry the candidate's local id within the cluster.
package hnsw

// Entry is a (distance, id) pair used by both heaps.
type Entry struct {
	Dist int32
	ID   uint16
}

// MinHeap is the candidate frontier: pop the closest unexpanded node next.
type MinHeap struct{ data []Entry }

func (h *MinHeap) Len() int    { return len(h.data) }
func (h *MinHeap) Reset()      { h.data = h.data[:0] }
func (h *MinHeap) Peek() Entry { return h.data[0] }

func (h *MinHeap) Push(e Entry) {
	h.data = append(h.data, e)
	i := len(h.data) - 1
	for i > 0 {
		p := (i - 1) / 2
		if h.data[p].Dist <= h.data[i].Dist {
			return
		}
		h.data[p], h.data[i] = h.data[i], h.data[p]
		i = p
	}
}

func (h *MinHeap) Pop() Entry {
	top := h.data[0]
	n := len(h.data) - 1
	h.data[0] = h.data[n]
	h.data = h.data[:n]
	i := 0
	for {
		l := 2*i + 1
		if l >= n {
			break
		}
		r := l + 1
		s := l
		if r < n && h.data[r].Dist < h.data[l].Dist {
			s = r
		}
		if h.data[i].Dist <= h.data[s].Dist {
			break
		}
		h.data[i], h.data[s] = h.data[s], h.data[i]
		i = s
	}
	return top
}

// MaxHeap is the result set: pop the *farthest* of the current best when
// we need to evict.
type MaxHeap struct{ data []Entry }

func (h *MaxHeap) Len() int       { return len(h.data) }
func (h *MaxHeap) Reset()         { h.data = h.data[:0] }
func (h *MaxHeap) Top() Entry     { return h.data[0] }
func (h *MaxHeap) Items() []Entry { return h.data }

func (h *MaxHeap) Push(e Entry) {
	h.data = append(h.data, e)
	i := len(h.data) - 1
	for i > 0 {
		p := (i - 1) / 2
		if h.data[p].Dist >= h.data[i].Dist {
			return
		}
		h.data[p], h.data[i] = h.data[i], h.data[p]
		i = p
	}
}

func (h *MaxHeap) Pop() Entry {
	top := h.data[0]
	n := len(h.data) - 1
	h.data[0] = h.data[n]
	h.data = h.data[:n]
	i := 0
	for {
		l := 2*i + 1
		if l >= n {
			break
		}
		r := l + 1
		s := l
		if r < n && h.data[r].Dist > h.data[l].Dist {
			s = r
		}
		if h.data[i].Dist >= h.data[s].Dist {
			break
		}
		h.data[i], h.data[s] = h.data[s], h.data[i]
		i = s
	}
	return top
}
