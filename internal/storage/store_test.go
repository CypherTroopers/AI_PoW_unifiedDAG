package storage

import "testing"

func TestContentAddressedStore(t *testing.T) {
	store, err := New(t.TempDir(), 1024)
	if err != nil {
		t.Fatal(err)
	}
	id, err := store.Put([]byte("AISeal proof"))
	if err != nil {
		t.Fatal(err)
	}
	data, ok := store.Get(id)
	if !ok || string(data) != "AISeal proof" {
		t.Fatal("stored object not retrievable")
	}
	reloaded, _ := New(store.dir, 1024)
	data, ok = reloaded.Get(id)
	if !ok || string(data) != "AISeal proof" {
		t.Fatal("persisted object failed content verification")
	}
}
