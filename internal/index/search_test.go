package index

import "testing"

func TestTop5_InsertSortedAscending(t *testing.T) {
	var top Top5
	// Insert out of order.
	top.insert(50, LabelLegit)
	top.insert(20, LabelFraud)
	top.insert(40, LabelLegit)
	top.insert(10, LabelFraud)
	top.insert(30, LabelLegit)
	top.insert(60, LabelFraud) // shouldn't displace (worse than current top)
	top.insert(5, LabelFraud)  // should displace 50

	if top.N != 5 {
		t.Fatalf("expected 5 entries, got %d", top.N)
	}
	want := []int32{5, 10, 20, 30, 40}
	for i, w := range want {
		if top.Dist[i] != w {
			t.Errorf("dist[%d] = %d, want %d", i, top.Dist[i], w)
		}
	}
	if got := top.FraudCount(); got != 3 {
		t.Errorf("fraud count = %d, want 3", got)
	}
}

func TestSelectTopK_PartialOrder(t *testing.T) {
	arr := []CentroidDist{
		{0, 5}, {1, 1}, {2, 8}, {3, 3}, {4, 7}, {5, 2}, {6, 9}, {7, 4},
	}
	selectTopK(arr, 3)
	// arr[:3] must contain the three smallest distances: {1, 2, 3}.
	seen := map[float32]bool{}
	for i := 0; i < 3; i++ {
		seen[arr[i].Dist] = true
	}
	for _, want := range []float32{1, 2, 3} {
		if !seen[want] {
			t.Errorf("missing %f in top-3 partition: %v", want, arr[:3])
		}
	}
}
