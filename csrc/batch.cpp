#include "batch.h"

#include <stdexcept>
#include <string>

namespace bwr {

Batch::Batch(int32_t capacity, int32_t n_seq_max_per_token) {
    if (capacity <= 0) {
        throw std::invalid_argument("Batch capacity must be >= 1, got " + std::to_string(capacity));
    }
    if (n_seq_max_per_token <= 0) {
        throw std::invalid_argument("Batch n_seq_max_per_token must be >= 1, got " +
                                    std::to_string(n_seq_max_per_token));
    }

    // embd = 0 -> token-id batch (the .token array is allocated, .embd stays null).
    batch_ = llama_batch_init(capacity, /*embd =*/ 0, n_seq_max_per_token);
    if (batch_.token == nullptr || batch_.pos == nullptr || batch_.n_seq_id == nullptr ||
        batch_.seq_id == nullptr || batch_.logits == nullptr) {
        llama_batch_free(batch_);
        throw std::runtime_error("llama_batch_init failed to allocate a batch of " +
                                 std::to_string(capacity) + " tokens");
    }

    capacity_            = capacity;
    n_seq_max_per_token_ = n_seq_max_per_token;
    batch_.n_tokens      = 0;
}

Batch::~Batch() {
    // llama_batch_free walks batch_.seq_id until the nullptr terminator that
    // llama_batch_init writes at index [capacity], so it must see the struct exactly
    // as init returned it -- clear() only ever touches n_tokens.
    llama_batch_free(batch_);
}

void Batch::clear() {
    batch_.n_tokens = 0;
    max_seq_id_     = -1;
    max_pos_        = -1;
}

int32_t Batch::add(llama_token token, llama_pos pos, llama_seq_id seq_id, bool logits) {
    const llama_seq_id one[1] = { seq_id };
    return add_shared(token, pos, std::vector<llama_seq_id>(one, one + 1), logits);
}

int32_t Batch::add_shared(llama_token token, llama_pos pos,
                          const std::vector<llama_seq_id> & seq_ids, bool logits) {
    if (batch_.n_tokens >= capacity_) {
        throw std::length_error("batch is full: capacity " + std::to_string(capacity_) +
                                " tokens already used; start a new step");
    }
    if (seq_ids.empty()) {
        throw std::invalid_argument("a batch token must belong to at least one sequence");
    }
    if ((int32_t) seq_ids.size() > n_seq_max_per_token_) {
        throw std::invalid_argument("token assigned to " + std::to_string(seq_ids.size()) +
                                    " sequences, batch allows " +
                                    std::to_string(n_seq_max_per_token_) + " per token");
    }
    if (pos < 0) {
        throw std::invalid_argument("batch position must be >= 0, got " + std::to_string(pos));
    }
    for (const llama_seq_id sid : seq_ids) {
        if (sid < 0) {
            throw std::invalid_argument("seq_id must be >= 0, got " + std::to_string(sid));
        }
    }

    const int32_t i = batch_.n_tokens;

    batch_.token[i]    = token;
    batch_.pos[i]      = pos;
    batch_.n_seq_id[i] = (int32_t) seq_ids.size();
    for (size_t s = 0; s < seq_ids.size(); ++s) {
        batch_.seq_id[i][s] = seq_ids[s];
        if (seq_ids[s] > max_seq_id_) {
            max_seq_id_ = seq_ids[s];
        }
    }
    batch_.logits[i]   = logits ? 1 : 0;

    if (pos > max_pos_) {
        max_pos_ = pos;
    }

    batch_.n_tokens = i + 1;
    return i;
}

} // namespace bwr
