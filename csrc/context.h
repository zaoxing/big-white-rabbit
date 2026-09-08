// Context handle: owns a llama_context (KV cache + compute) and a sampler chain.
//
// Phase 0 scope: single-sequence decode via llama_batch_get_one, so the bring-up path
// stays as short as possible. Phase 1 replaces decode_seq0() with the real llama_batch
// path (mixed prefill + multi-sequence decode) in batch.*, at which point this class
// keeps the context/sampler ownership and hands batching to MetalEngine.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "llama.h"
#include "model.h"

namespace ftm {

struct ContextParams {
    uint32_t n_ctx           = 4096;
    uint32_t n_batch         = 512;
    uint32_t n_ubatch        = 512;
    uint32_t n_seq_max       = 1;
    int32_t  n_threads       = 0;  // 0 -> llama default (physical core count)
    int32_t  n_threads_batch = 0;
    bool     flash_attn      = true;
};

struct SamplerParams {
    float    temp  = 0.0f;  // <= 0 -> greedy
    int32_t  top_k = 40;
    float    top_p = 0.95f;
    uint32_t seed  = LLAMA_DEFAULT_SEED;
};

class Context {
public:
    Context(std::shared_ptr<Model> model, const ContextParams & cp, const SamplerParams & sp);
    ~Context();

    Context(const Context &)             = delete;
    Context & operator=(const Context &) = delete;

    // Decode a run of tokens into sequence 0. Positions are tracked internally by
    // llama_decode. Throws on a KV-slot miss (return 1) or a fatal decode error (< 0).
    void decode_seq0(const std::vector<llama_token> & tokens);

    // Sample from the logits of the last decoded token.
    llama_token sample_last();

    // Accept a token into the sampler chain's state (needed once penalties/grammar
    // samplers enter the chain; harmless for greedy).
    void accept(llama_token tok);

    // Drop sequence `seq_id`'s KV from `p0` (inclusive) to `p1` (exclusive);
    // negative bounds mean "open ended". This is the primitive the Phase 4 radix
    // cache is built on (llama_memory_seq_rm, formerly llama_kv_cache_seq_rm).
    void memory_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1);

    // Effective geometry, post-clamping. llama.cpp rounds n_ctx UP (KV padding) and
    // clamps n_batch DOWN to the requested n_ctx, so neither necessarily matches what
    // was asked for -- callers must budget against these, not against their own request.
    uint32_t n_ctx()    const;
    uint32_t n_batch()  const;
    uint32_t n_ubatch() const;

private:
    std::shared_ptr<Model> model_;
    llama_context * ctx_  = nullptr;
    llama_sampler * smpl_ = nullptr;
};

} // namespace ftm
