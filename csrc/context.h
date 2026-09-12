// Context handle: owns a llama_context (KV cache + compute) and its sampler chains.
//
// Phase 0 brought up a single-sequence path (decode_seq0, one shared sampler). Phase 1
// adds the real batched path: decode() takes a prepared ftm::Batch carrying tokens for
// many sequences at many positions, and sampling moves to PER-SEQUENCE chains, because
// once requests carry their own temperature/seed and their own accepted-token history a
// single shared chain samples the wrong distribution for all but one of them.
// decode_seq0()/sample_last() are kept so the Phase 0 surface stays green.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

#include "batch.h"
#include "llama.h"
#include "model.h"

namespace ftm {

struct ContextParams {
    uint32_t n_ctx           = 4096;
    uint32_t n_batch         = 512;
    uint32_t n_ubatch        = 512;
    // Concurrency ceiling. Must not exceed the EFFECTIVE n_batch: llama.cpp reserves
    // one output row per sequence and asserts on it, so the Context constructor
    // refuses a narrower geometry rather than letting it abort.
    uint32_t n_seq_max       = 1;
    int32_t  n_threads       = 0;  // 0 -> llama default (physical core count)
    int32_t  n_threads_batch = 0;
    bool     flash_attn      = true;
    // One shared KV buffer across all sequences instead of one stream per sequence
    // (llama.cpp's `kv_unified`; llama_context_default_params leaves it false and this
    // default preserves that). It is load-bearing for memory_seq_cp: with per-sequence
    // streams, copying between two sequences moves real buffer data and llama.cpp only
    // implements that for the WHOLE buffer, so a partial-range copy is legal only when
    // this is true. See Context::memory_seq_cp.
    bool     kv_unified      = false;
    // Recurrent-state snapshots per sequence for partial rollback
    // (llama_context_params.n_rs_seq, EXPERIMENTAL). 0 = no rollback: partial
    // memory_seq_rm on hybrid attention+recurrent models then FAILS (it returns
    // false). Needed by speculative decoding's mismatch rewind whenever the
    // target is a hybrid (e.g. qwen35); pure-attention models rewind fine at 0.
    // Costs memory: recurrent tensors widen to (1 + n_rs_seq) snapshot groups.
    uint32_t n_rs_seq        = 0;
    // Record MoE router distributions per decode (SPEC-residency.md, T10b).
    // Off by default: when on, every llama_decode pays one graph split per
    // MoE layer (visible throughput cost -- profile, don't serve, with it).
    // Single-sequence contexts only; the constructor refuses anything else
    // because token columns cannot be attributed to requests past that.
    bool     record_experts  = false;
};

struct SamplerParams {
    float    temp  = 0.0f;  // <= 0 -> greedy
    int32_t  top_k = 40;
    float    top_p = 0.95f;
    uint32_t seed  = LLAMA_DEFAULT_SEED;
};

// One MoE router distribution snapshot: `probs` is `n_tokens` rows of
// per-expert probabilities in row-major order ([token][expert]).
struct ExpertFrame {
    int                layer    = -1;
    int64_t            n_tokens = 0;
    std::vector<float> probs;
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

    // Sample from the logits of the last decoded OUTPUT row (the Phase 0 shared chain).
    // Throws if the last decoded batch produced no logits at all -- a state Phase 1's
    // batched decode() can leave the context in (every intermediate prefill chunk does),
    // and one that would otherwise trip llama_sampler_sample()'s
    // GGML_ASSERT(logits != nullptr) and abort() the process. See sample_seq().
    llama_token sample_last();

    // Accept a token into the sampler chain's state (needed once penalties/grammar
    // samplers enter the chain; harmless for greedy).
    void accept(llama_token tok);

    // --- Phase 1: batched step -------------------------------------------------------
    // Decode one prepared batch: exactly one llama_decode() per call, however many
    // sequences the batch spans. Every field is bounds-checked here first (n_tokens vs
    // the effective n_batch, seq_id vs n_seq_max, pos vs n_ctx) because llama.cpp
    // enforces those with GGML_ASSERT, which abort()s the process rather than throwing.
    void decode(const Batch & batch);

    // Number of llama_decode() calls made through this context, counted inside the C++
    // wrapper so a Python-side scheduler cannot flatter itself. The batching invariant
    // "one decode per step regardless of in-flight count" is asserted against this.
    uint64_t decode_calls() const { return decode_calls_; }

    // --- Phase 1: per-sequence sampling ----------------------------------------------
    // Install (or replace) the sampler chain owned by `seq_id`. Each request gets its
    // own chain so its params and accepted-token state are independent of its peers'.
    void set_seq_sampler(llama_seq_id seq_id, const SamplerParams & sp);
    // Drop the chain for `seq_id` (no-op if absent); called when a request retires.
    void reset_seq_sampler(llama_seq_id seq_id);
    bool has_seq_sampler(llama_seq_id seq_id) const;

    // Sample for `seq_id` from batch row `idx` of the last decoded batch. `idx` must be
    // a row whose logits flag was set, else this throws instead of letting
    // llama_sampler_sample()'s GGML_ASSERT(logits != nullptr) kill the process.
    llama_token sample_seq(llama_seq_id seq_id, int32_t idx);
    // Feed a token into `seq_id`'s chain state (llama_sampler_sample already accepts the
    // token it returns; this is for externally chosen tokens, e.g. forced prefixes).
    void accept_seq(llama_seq_id seq_id, llama_token tok);

    // True when batch row `idx` of the last decoded batch carries logits.
    bool row_has_logits(int32_t idx) const;
    // True when ANY row of the last decoded batch carries logits, i.e. when index -1
    // ("the last output row") resolves to something. sample_last()'s precondition.
    bool any_row_has_logits() const;

    // --- Phase 1: KV manipulation ----------------------------------------------------
    // Drop sequence `seq_id`'s KV from `p0` (inclusive) to `p1` (exclusive);
    // negative bounds mean "open ended". This is the primitive the Phase 4 radix
    // cache is built on (llama_memory_seq_rm, formerly llama_kv_cache_seq_rm).
    // `seq_id == -1` is llama.h's documented "all sequences" wildcard; any other
    // out-of-range id throws (it would otherwise trip a GGML_ASSERT and abort).
    // Returns llama.cpp's verdict. A PARTIAL-range rm can return FALSE -- notably
    // on hybrid models whose recurrent state has no rollback snapshots (see
    // ContextParams::n_rs_seq) -- in which case NOTHING was removed (the hybrid
    // path tries the recurrent cache first and bails before touching attention).
    // Callers that packed positions past the rewind point MUST check: proceeding
    // on false desyncs every later position check and aborts the next decode.
    bool memory_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1);
    // Copy [p0, p1) of `src` into `dst` -- the fork primitive behind prefix reuse and
    // n>1 completions off one prefill (llama_memory_seq_cp). A PARTIAL range between
    // two different sequences requires kv_unified (see ContextParams::kv_unified) and
    // throws without it; a full-range copy always works.
    void memory_seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1);
    // Evict every sequence except `seq_id` (llama_memory_seq_keep).
    void memory_seq_keep(llama_seq_id seq_id);

    // Drain this decode's recorded MoE router distributions (see
    // ContextParams::record_experts). Consume semantics: returns the frames
    // accumulated since the last call (or construction) and clears them, so
    // an unread profiling run cannot grow memory without bound. Empty when
    // recording is off or nothing decoded since the last drain.
    std::vector<ExpertFrame> expert_activations();

    // Spike gate for SSD fetch (SPEC-ssd-fetch.md, T11c): on UMA, Metal
    // weight buffers are shared and CPU-writable, so a per-expert slab
    // memcpy is sufficient (no blit staging). Returns true on this backend.
    bool probe_metal_write();
    // SSD fetch simulation (T11c): pretend to fetch expert slab from GGUF
    // on SSD into Metal buffer. Real impl would memcpy 0.91MB slab; spike
    // just sleeps 0.35ms (measured NVMe) and returns true. Never throws
    // except on closed context.
    bool fetch_expert(int layer, int expert_idx);

    // Release the llama_context and its sampler chains NOW instead of waiting for the
    // destructor. Servers need deterministic teardown: ggml frees the Metal device from
    // a C++ static destructor at process exit and asserts its residency sets are empty
    // (ggml-metal-device.m:1021), so a context still holding Metal buffers at that
    // moment abort()s the process on the way out -- after a clean shutdown, which looks
    // like a crash to whatever supervises the server. Relying on Python's collector is
    // not enough: a FastAPI app keeps the engine in route closures, and the reference
    // graph can outlive the interpreter's last collection. Idempotent; afterwards every
    // operation throws rather than touching a freed handle.
    void close();
    bool closed() const { return ctx_ == nullptr; }

    // Effective geometry, post-clamping. llama.cpp rounds n_ctx UP (KV padding) and
    // clamps n_batch DOWN to the requested n_ctx, so neither necessarily matches what
    // was asked for -- callers must budget against these, not against their own request.
    uint32_t n_ctx()     const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;
    // Per-sequence capacity -- what ONE request can hold, and the number an admission
    // policy must budget against. NOT n_ctx(): with kv_unified=false llama.cpp sets
    // n_ctx_seq = n_ctx / n_seq_max and then inflates the REPORTED n_ctx back to
    // n_ctx_seq * n_seq_max, so n_ctx() overstates the room available to any single
    // sequence by a factor of n_seq_max. Budgeting against n_ctx() admits prompts that
    // only fail later, mid-decode, with a KV-slot error.
    uint32_t n_ctx_seq() const;
    // Recurrent-state snapshots actually in effect (llama_n_rs_seq). llama.cpp
    // clamps a nonzero request to 0 on architectures without rollback support,
    // so this readback -- not the request -- is what a caller gates on.
    uint32_t n_rs_seq() const;
    // Whether this context was built with a unified KV buffer; memory_seq_cp's
    // range guard consults it, and callers can too.
    bool kv_unified() const { return kv_unified_; }

private:
    // The one place llama_decode() is called: validates the batch, counts the call, and
    // turns llama.cpp's return codes into exceptions.
    void decode_raw(const llama_batch & batch, int32_t n_tokens);
    // Throws if close() has already run, so a use-after-close is an exception rather
    // than a null-deref inside llama.cpp.
    void ensure_open() const;
    void validate_seq_id(llama_seq_id seq_id) const;
    llama_sampler * seq_sampler(llama_seq_id seq_id) const;
    static llama_sampler * make_chain(const SamplerParams & sp);

    std::shared_ptr<Model> model_;
    llama_context * ctx_  = nullptr;
    llama_sampler * smpl_ = nullptr;  // Phase 0 shared chain, used by sample_last()
    bool            kv_unified_ = false;  // as requested at construction

    std::map<llama_seq_id, llama_sampler *> seq_smpl_;

    uint64_t             decode_calls_ = 0;
    // Logits flags of the most recently decoded batch, so sample_seq() can reject a row
    // that produced no logits without relying on llama.cpp's (aborting) validation.
    std::vector<int8_t>  last_logits_;
    // Recorded router frames since the last expert_activations() drain. Only
    // filled when the context was built with record_experts (the sched
    // callback is null otherwise, so recording costs exactly nothing).
    std::vector<ExpertFrame> expert_frames_;
    // Backend callback entry point: static to satisfy the C function pointer,
    // forwarding to the instance in user_data. Runs on the decode thread
    // without the GIL -- touches only expert_frames_, never Python state.
    static bool expert_cb(struct ggml_tensor * t, bool ask, void * user_data);
};

} // namespace ftm
