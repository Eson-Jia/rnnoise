package rnnoise

/*
#cgo CFLAGS: -I../include -I../src -DRNNOISE_EXPORT= -DUSE_WEIGHTS_FILE
#cgo LDFLAGS: -lm
#include <stdlib.h>
#include "../include/rnnoise.h"
#include "../src/denoise.c"
#include "../src/rnn.c"
#include "../src/pitch.c"
#include "../src/kiss_fft.c"
#include "../src/celt_lpc.c"
#include "../src/nnet.c"
#include "../src/nnet_default.c"
#include "../src/parse_lpcnet_weights.c"
#include "../src/rnnoise_tables.c"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// RNNoise wraps a DenoiseState pointer.
type RNNoise struct {
	state *C.DenoiseState
	model *C.RNNModel
}

// New loads a model from the provided path and creates a RNNoise instance.
func New(modelPath string) (*RNNoise, error) {
	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))
	model := C.rnnoise_model_from_filename(cpath)
	if model == nil {
		return nil, fmt.Errorf("failed to load model: %s", modelPath)
	}
	st := C.rnnoise_create(model)
	if st == nil {
		C.rnnoise_model_free(model)
		return nil, fmt.Errorf("rnnoise_create failed")
	}
	return &RNNoise{state: st, model: model}, nil
}

// Close releases resources held by the RNNoise instance.
func (r *RNNoise) Close() {
	if r.state != nil {
		C.rnnoise_destroy(r.state)
		r.state = nil
	}
	if r.model != nil {
		C.rnnoise_model_free(r.model)
		r.model = nil
	}
}

// FrameSize returns the number of samples processed per frame.
func FrameSize() int {
	return int(C.rnnoise_get_frame_size())
}

// Process applies noise suppression to the given 16-bit PCM frame in-place.
// The slice length must be at least FrameSize().
func (r *RNNoise) Process(frame []int16) {
	n := C.rnnoise_get_frame_size()
	if len(frame) < int(n) || r.state == nil {
		return
	}
	buf := make([]C.float, n)
	for i := 0; i < int(n); i++ {
		buf[i] = C.float(frame[i])
	}
	C.rnnoise_process_frame(r.state, &buf[0], &buf[0])
	for i := 0; i < int(n); i++ {
		frame[i] = int16(buf[i])
	}
}

// ProcessFloat32 applies noise suppression to a float32 frame in-place.
func (r *RNNoise) ProcessFloat32(frame []float32) {
	n := C.rnnoise_get_frame_size()
	if len(frame) < int(n) || r.state == nil {
		return
	}
	ptr := (*C.float)(unsafe.Pointer(&frame[0]))
	C.rnnoise_process_frame(r.state, ptr, ptr)
}
