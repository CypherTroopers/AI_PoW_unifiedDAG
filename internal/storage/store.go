package storage

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

type Store struct {
	mu       sync.RWMutex
	objects  map[string][]byte
	dir      string
	maxBytes int
}

func New(dir string, maxBytes int) (*Store, error) {
	if maxBytes <= 0 {
		maxBytes = 16 << 20
	}
	if dir != "" {
		if err := os.MkdirAll(dir, 0o750); err != nil {
			return nil, err
		}
	}
	return &Store{objects: make(map[string][]byte), dir: dir, maxBytes: maxBytes}, nil
}

func (s *Store) Put(data []byte) (string, error) {
	if len(data) == 0 || len(data) > s.maxBytes {
		return "", errors.New("object is empty or exceeds storage limit")
	}
	hash := sha256.Sum256(data)
	id := "0x" + hex.EncodeToString(hash[:])
	copyData := append([]byte(nil), data...)
	s.mu.Lock()
	s.objects[id] = copyData
	s.mu.Unlock()
	if s.dir != "" {
		path := filepath.Join(s.dir, id[2:])
		if err := writeAtomic(path, copyData); err != nil {
			return "", err
		}
	}
	return id, nil
}

func (s *Store) Get(id string) ([]byte, bool) {
	id = normalizeID(id)
	s.mu.RLock()
	data, ok := s.objects[id]
	s.mu.RUnlock()
	if ok {
		return append([]byte(nil), data...), true
	}
	if s.dir == "" || !validID(id) {
		return nil, false
	}
	data, err := os.ReadFile(filepath.Join(s.dir, id[2:]))
	if err != nil || len(data) > s.maxBytes {
		return nil, false
	}
	hash := sha256.Sum256(data)
	if id != "0x"+hex.EncodeToString(hash[:]) {
		return nil, false
	}
	s.mu.Lock()
	s.objects[id] = append([]byte(nil), data...)
	s.mu.Unlock()
	return data, true
}

func normalizeID(id string) string {
	id = strings.ToLower(strings.TrimSpace(id))
	if !strings.HasPrefix(id, "0x") {
		id = "0x" + id
	}
	return id
}

func validID(id string) bool {
	if len(id) != 66 || !strings.HasPrefix(id, "0x") {
		return false
	}
	_, err := hex.DecodeString(id[2:])
	return err == nil
}

func writeAtomic(path string, data []byte) error {
	temp, err := os.CreateTemp(filepath.Dir(path), ".object-*")
	if err != nil {
		return err
	}
	tempPath := temp.Name()
	defer os.Remove(tempPath)
	if _, err := temp.Write(data); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	return os.Rename(tempPath, path)
}
