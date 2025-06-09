package rnnoise

import (
	"os"
	"testing"
)

func TestRNNoise(t *testing.T) {
	model := os.Getenv("RNNOISE_MODEL")
	if model == "" {
		t.Skip("RNNOISE_MODEL not set")
	}
	r, err := New(model)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	size := FrameSize()
	buf := make([]int16, size)
	r.Process(buf)
}
