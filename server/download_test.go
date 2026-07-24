package server

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

func BenchmarkDownloadChunkCompletion(b *testing.B) {
	data := make([]byte, 1024*1024)
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprint(len(data)))
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(data)
	}))
	b.Cleanup(server.Close)

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		b.Fatal(err)
	}
	downloadPath := filepath.Join(b.TempDir(), "blob")

	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		download := &blobDownload{Name: downloadPath, Digest: digest}
		part := &blobDownloadPart{Size: int64(len(data)), blobDownload: download}
		if err := download.downloadChunk(b.Context(), requestURL, io.Discard, part); err != nil {
			b.Fatal(err)
		}
	}
}

func TestDownloadChunkReturnsWhenTransferCompletes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1")
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write([]byte{0})
	}))
	t.Cleanup(server.Close)

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}

	download := &blobDownload{
		Name:   filepath.Join(t.TempDir(), "blob"),
		Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000",
	}
	part := &blobDownloadPart{Size: 1, blobDownload: download}
	ctx, cancel := context.WithTimeout(t.Context(), 250*time.Millisecond)
	defer cancel()

	if err := download.downloadChunk(ctx, requestURL, io.Discard, part); err != nil {
		t.Fatalf("downloadChunk() error = %v, want nil", err)
	}
}

func TestDownloadChunkDetectsStallBeforeFirstByte(t *testing.T) {
	requestStarted := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(requestStarted)
		w.Header().Set("Content-Length", "1")
		w.WriteHeader(http.StatusPartialContent)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(server.Close)

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}

	download := &blobDownload{Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000"}
	part := &blobDownloadPart{Size: 1, blobDownload: download}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()

	originalStallTimeout := downloadStallTimeout
	downloadStallTimeout = 50 * time.Millisecond
	t.Cleanup(func() {
		downloadStallTimeout = originalStallTimeout
	})

	started := time.Now()
	err = download.downloadChunk(ctx, requestURL, io.Discard, part)
	elapsed := time.Since(started)

	select {
	case <-requestStarted:
	default:
		t.Fatal("download request did not start")
	}
	if !errors.Is(err, errPartStalled) {
		t.Fatalf("downloadChunk() error = %v after %v, want %v", err, elapsed, errPartStalled)
	}
	if elapsed >= 5*downloadStallTimeout {
		t.Fatalf("downloadChunk() detected the stall after %v, want less than %v", elapsed, 5*downloadStallTimeout)
	}
}

// ---------------------------------------------------------------------------
// Phase 1: velocity tracking
// ---------------------------------------------------------------------------

func TestVelocityWindow_AccumulatesBytes(t *testing.T) {
	download := &blobDownload{Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000"}
	part := &blobDownloadPart{blobDownload: download}

	// Write 10 KB — all within the rolling window.
	data := make([]byte, 10*1024)
	if _, err := part.Write(data); err != nil {
		t.Fatal(err)
	}

	bps := part.currentBytesPerSec()
	if bps <= 0 {
		t.Fatalf("currentBytesPerSec() = %v, want > 0", bps)
	}
}

func TestVelocityWindow_EvictsOldSamples(t *testing.T) {
	download := &blobDownload{Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000"}
	part := &blobDownloadPart{blobDownload: download}

	// Inject an old sample directly (older than the rolling window).
	old := velocitySample{
		t:     time.Now().Add(-(velocityWindowDuration + time.Second)),
		bytes: 1024 * 1024, // 1 MB — should not be counted
	}
	part.velocityMu.Lock()
	part.velocitySamples = []velocitySample{old}
	part.velocityMu.Unlock()

	// Write a fresh tiny payload.
	if _, err := part.Write([]byte{1}); err != nil {
		t.Fatal(err)
	}

	bps := part.currentBytesPerSec()
	// The 1 MB old sample must NOT skew the result above ~1 KB/s.
	if bps > 1024 {
		t.Fatalf("currentBytesPerSec() = %.1f, old samples should have been evicted", bps)
	}
}

// ---------------------------------------------------------------------------
// Phase 2: adaptive stall timeout
// ---------------------------------------------------------------------------

func TestAdaptiveStallTimeout(t *testing.T) {
	cases := []struct {
		bps  float64
		want time.Duration
	}{
		{bps: 200 * 1024, want: 30 * time.Second},  // fast
		{bps: 50 * 1024, want: 60 * time.Second},   // medium
		{bps: 5 * 1024, want: 120 * time.Second},   // slow
		{bps: 0, want: 120 * time.Second},           // zero (no data yet)
	}
	for _, c := range cases {
		got := adaptiveStallTimeout(c.bps)
		if got != c.want {
			t.Errorf("adaptiveStallTimeout(%.0f) = %v, want %v", c.bps, got, c.want)
		}
	}
}

// ---------------------------------------------------------------------------
// Phase 3: adaptive part count
// ---------------------------------------------------------------------------

func TestAdaptivePartCount(t *testing.T) {
	cases := []struct {
		bps  float64
		want int
	}{
		{bps: 600 * 1024, want: 16},
		{bps: 100 * 1024, want: 8},
		{bps: 20 * 1024, want: 4},
		{bps: 0, want: 4},
	}
	for _, c := range cases {
		got := adaptivePartCount(c.bps)
		if got != c.want {
			t.Errorf("adaptivePartCount(%.0f) = %d, want %d", c.bps, got, c.want)
		}
	}
}

// ---------------------------------------------------------------------------
// Phase 4: multi-stage stall detection
// ---------------------------------------------------------------------------

// TestStagedStall_FirstStallWarnsOnly verifies that the first stall does NOT
// return errPartStalled — it extends the grace period and continues.
func TestStagedStall_FirstStallWarnsOnly(t *testing.T) {
	var serveCount atomic.Int32
	// The server stalls once and then delivers data on the second connection.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := serveCount.Add(1)
		w.Header().Set("Content-Length", "1")
		w.WriteHeader(http.StatusPartialContent)
		w.(http.Flusher).Flush()
		if n == 1 {
			// First request: stall until the client disconnects.
			<-r.Context().Done()
			return
		}
		// Second+ request: deliver the byte.
		_, _ = w.Write([]byte{0})
	}))
	t.Cleanup(server.Close)

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}

	originalStallTimeout := downloadStallTimeout
	downloadStallTimeout = 40 * time.Millisecond
	t.Cleanup(func() { downloadStallTimeout = originalStallTimeout })

	download := &blobDownload{Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000"}
	part := &blobDownloadPart{Size: 1, blobDownload: download}

	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()

	// First downloadChunk call: stall counter goes to 1 (warn), then to 2
	// (reconnect) → returns errPartStalled.
	err = download.downloadChunk(ctx, requestURL, io.Discard, part)
	if !errors.Is(err, errPartStalled) {
		t.Fatalf("expected errPartStalled after two stall stages, got %v", err)
	}
	if part.stallCount.Load() < 2 {
		t.Fatalf("stallCount = %d, want >= 2", part.stallCount.Load())
	}
}

// TestStagedStall_MaxStallRetriesExceeded verifies the outer retry loop in
// run() stops after maxStallRetries stall-reconnects.
func TestStagedStall_MaxStallRetriesExceeded(t *testing.T) {
	// Server always stalls — never sends data.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1")
		w.WriteHeader(http.StatusPartialContent)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(server.Close)

	originalStallTimeout := downloadStallTimeout
	downloadStallTimeout = 30 * time.Millisecond
	t.Cleanup(func() { downloadStallTimeout = originalStallTimeout })

	download := &blobDownload{Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000"}
	part := &blobDownloadPart{Size: 1, blobDownload: download}

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}

	stallRetries := 0
	for i := 0; i < maxStallRetries+2; i++ {
		part.stallCount.Store(0)
		ctx, cancel := context.WithTimeout(t.Context(), 500*time.Millisecond)
		err = download.downloadChunk(ctx, requestURL, io.Discard, part)
		cancel()
		if errors.Is(err, errPartStalled) {
			stallRetries++
			if stallRetries >= maxStallRetries {
				break
			}
		}
	}
	if stallRetries < maxStallRetries {
		t.Fatalf("stallRetries = %d, want >= %d", stallRetries, maxStallRetries)
	}
}
