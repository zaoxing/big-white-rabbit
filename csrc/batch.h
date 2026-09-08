// RAII wrapper around llama_batch.
//
// llama_batch_init() hands back a struct of raw malloc'd arrays that must be released
// with llama_batch_free(). Every member is left for the caller to fill in, and nothing
// in the C API bounds-checks the writes: overrunning `n_tokens` past the allocation is a
// silent heap corruption, and a throw between init and free is a leak. Batch owns the
// allocation for its whole lifetime (freed in the destructor, so an exception on any
// append path cannot leak it) and range-checks every append.
//
// This is the Phase 1 replacement for Phase 0's llama_batch_get_one() path: one batch
// can now carry tokens for many sequences at different positions, which is what lets N
// requests interleave through a single llama_decode() call.
#pragma once

#include <cstdint>
#include <vector>

#include "llama.h"

namespace ftm {

class Batch {
public:
    // `capacity` tokens, each able to carry up to `n_seq_max_per_token` sequence ids
    // (that is llama_batch_init's third argument: the per-token seq-id fan-out for
    // shared prefixes, NOT the context's n_seq_max).
    Batch(int32_t capacity, int32_t n_seq_max_per_token = 1);
    ~Batch();

    Batch(const Batch &)             = delete;
    Batch & operator=(const Batch &) = delete;

    // Reset to empty without reallocating; the engine reuses one batch per step.
    void clear();

    // Append one token. Returns its row index in the batch -- that index is what
    // llama_get_logits_ith()/llama_sampler_sample() take to read this token's logits.
    // Throws std::length_error past capacity and std::invalid_argument on a bad
    // pos/seq_id, so an oversized or malformed step cannot reach llama.cpp's asserts.
    int32_t add(llama_token token, llama_pos pos, llama_seq_id seq_id, bool logits);

    // Same, but the token is written into several sequences at once (prefix sharing).
    int32_t add_shared(llama_token token, llama_pos pos,
                       const std::vector<llama_seq_id> & seq_ids, bool logits);

    int32_t n_tokens() const { return batch_.n_tokens; }
    int32_t capacity() const { return capacity_; }
    int32_t n_seq_max_per_token() const { return n_seq_max_per_token_; }

    // The largest seq_id written into the current contents, or -1 when empty.
    // Context::decode() uses this to reject out-of-range ids before llama.cpp does.
    llama_seq_id max_seq_id() const { return max_seq_id_; }
    // The largest position written into the current contents, or -1 when empty.
    llama_pos max_pos() const { return max_pos_; }

    // Non-const because llama_decode takes the struct by value and may read through
    // the pointers; callers must not resize or free anything reached from here.
    const llama_batch & raw() const { return batch_; }

private:
    llama_batch  batch_{};
    int32_t      capacity_             = 0;
    int32_t      n_seq_max_per_token_  = 0;
    llama_seq_id max_seq_id_           = -1;
    llama_pos    max_pos_              = -1;
};

} // namespace ftm
