#pragma once

// Backend-agnostic LRU slot manager for a persistent pool of MoE expert-weight
// slots living in dedicated VRAM (separate from the graph allocator, so the
// resident weights survive across forward passes). This header holds only the
// bookkeeping: an opaque key (e.g. layer/projection/expert packed into a
// uint64) -> slot index, with plain LRU eviction (the policy chosen offline in
// tools/predictive-prefetch/simulate_cache.py, ~92-94% of the Belady optimum).
//
// The CUDA-specific part (cudaMalloc of the pool, device<->device / host->device
// copies into slots) wraps this. On a hit the caller copies VRAM->VRAM from the
// slot; on a miss the caller copies host->VRAM into the (possibly evicted) slot.
//
// O(1) per access: a hash map for lookup and an intrusive doubly-linked list for
// recency order (front = most-recently-used, back = least).

#include <cstdint>
#include <vector>
#include <unordered_map>

class expert_cache_lru {
public:
    static constexpr uint64_t KEY_NONE = ~0ull;

    struct access_result {
        int      slot;          // slot index assigned to the key
        bool     hit;           // key was already resident
        bool     evicted;       // a victim was evicted to make room (only on miss)
        uint64_t evicted_key;   // the victim's key (valid iff evicted)
    };

    explicit expert_cache_lru(int n_slots) : slots_(n_slots) {
        map_.reserve(n_slots * 2);
        // build the free list and an initially empty LRU list
        for (int i = 0; i < n_slots; ++i) {
            slots_[i].key  = KEY_NONE;
            slots_[i].prev = -1;
            slots_[i].next = -1;
        }
        for (int i = n_slots - 1; i >= 0; --i) {
            free_.push_back(i);
        }
    }

    int  capacity() const { return (int) slots_.size(); }
    bool resident(uint64_t key) const { return map_.find(key) != map_.end(); }

    // Look up key; insert on miss. Returns the slot and whether it was a hit.
    access_result access(uint64_t key) {
        access_result r{};
        auto it = map_.find(key);
        if (it != map_.end()) {
            const int s = it->second;
            list_move_to_front(s);
            r.slot = s; r.hit = true; r.evicted = false;
            return r;
        }

        int s;
        if (!free_.empty()) {
            s = free_.back();
            free_.pop_back();
            r.evicted = false;
        } else {
            s = lru_tail_;              // least-recently-used
            r.evicted = true;
            r.evicted_key = slots_[s].key;
            map_.erase(slots_[s].key);
            list_unlink(s);
        }

        slots_[s].key = key;
        map_[key] = s;
        list_push_front(s);
        r.slot = s; r.hit = false;
        return r;
    }

private:
    struct slot {
        uint64_t key;
        int      prev;  // toward MRU
        int      next;  // toward LRU
    };

    void list_push_front(int s) {
        slots_[s].prev = -1;
        slots_[s].next = lru_head_;
        if (lru_head_ != -1) {
            slots_[lru_head_].prev = s;
        }
        lru_head_ = s;
        if (lru_tail_ == -1) {
            lru_tail_ = s;
        }
    }

    void list_unlink(int s) {
        const int p = slots_[s].prev;
        const int n = slots_[s].next;
        if (p != -1) slots_[p].next = n; else lru_head_ = n;
        if (n != -1) slots_[n].prev = p; else lru_tail_ = p;
        slots_[s].prev = slots_[s].next = -1;
    }

    void list_move_to_front(int s) {
        if (lru_head_ == s) {
            return;
        }
        list_unlink(s);
        list_push_front(s);
    }

    std::vector<slot>                 slots_;
    std::vector<int>                  free_;
    std::unordered_map<uint64_t, int> map_;
    int                               lru_head_ = -1;  // MRU
    int                               lru_tail_ = -1;  // LRU
};
