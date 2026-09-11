#include "context.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ftm {

Context::Context(std::shared_ptr<Model> model, const ContextParams & cp, const SamplerParams & sp)
    : model_(std::move(model)) {
    if (!model_) {
        throw std::invalid_argument("Context requires a loaded Model");
    }

    // llama.cpp leaves n_outputs_max at 0, which it resolves to the EFFECTIVE n_batch,
    // and llama_context::output_reserve() then asserts
    // `max(n_outputs, n_seq_max) <= n_outputs_max` -- one output row per sequence has to
    // fit. A geometry whose effective n_batch is narrower than n_seq_max therefore
    // abort()s INSIDE llama_init_from_model, before any handle exists to guard. Predict
    // the two clamps llama.cpp applies (n_ctx == 0 means the model's training context;
    // with causal attention n_batch is clamped down to the REQUESTED n_ctx) and refuse
    // the geometry as an exception instead. n_batch == 0 lands here too.
    const uint32_t n_seq       = cp.n_seq_max > 0 ? cp.n_seq_max : 1;
    const uint32_t req_n_ctx   = cp.n_ctx > 0 ? cp.n_ctx : (uint32_t) model_->n_ctx_train();
    const uint32_t eff_n_batch = cp.n_batch < req_n_ctx ? cp.n_batch : req_n_ctx;
    if (eff_n_batch < n_seq) {
        throw std::invalid_argument(
            "n_seq_max=" + std::to_string(n_seq) + " needs an effective n_batch of at least "
            "that many tokens (llama.cpp reserves one output row per sequence), but "
            "n_batch=" + std::to_string(cp.n_batch) + " clamped to n_ctx=" +
            std::to_string(req_n_ctx) + " gives " + std::to_string(eff_n_batch) +
            "; raise n_batch/n_ctx or lower n_seq_max");
    }

    llama_context_params lp = llama_context_default_params();
    lp.n_ctx           = cp.n_ctx;
    lp.n_batch         = cp.n_batch;
    lp.n_ubatch        = cp.n_ubatch;
    lp.n_seq_max       = cp.n_seq_max;
    lp.kv_unified      = cp.kv_unified;
    lp.n_rs_seq        = cp.n_rs_seq;
    lp.flash_attn_type = cp.flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO
                                       : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (cp.record_experts) {
        // Token columns cannot be attributed to requests past one sequence,
        // so multi-seq recording is refused loudly rather than misattributed
        // silently. (Single-seq covers profiling, tests, and the qstar bench.)
        if (cp.n_seq_max != 1) {
            throw std::invalid_argument(
                "record_experts needs n_seq_max == 1 (got " +
                std::to_string(cp.n_seq_max) + "); multi-sequence attribution "
                "is a later item, not a silent misattribution");
        }
        lp.cb_eval = &Context::expert_cb;
        lp.cb_eval_user_data = this;
    }
    if (cp.n_threads > 0) {
        lp.n_threads = cp.n_threads;
    }
    if (cp.n_threads_batch > 0) {
        lp.n_threads_batch = cp.n_threads_batch;
    }

    ctx_ = llama_init_from_model(model_->raw(), lp);
    if (ctx_ == nullptr) {
        throw std::runtime_error("failed to create llama_context");
    }
    // llama.cpp keeps this in its private cparams with no getter, so remember what we
    // asked for: memory_seq_cp's range guard depends on it.
    kv_unified_ = cp.kv_unified;

    try {
        smpl_ = make_chain(sp);
    } catch (...) {
        llama_free(ctx_);
        ctx_ = nullptr;
        throw;
    }
}

llama_sampler * Context::make_chain(const SamplerParams & sp) {
    llama_sampler_chain_params scp = llama_sampler_chain_default_params();
    llama_sampler * chain = llama_sampler_chain_init(scp);
    if (chain == nullptr) {
        throw std::runtime_error("failed to create sampler chain");
    }

    if (sp.temp <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        // Conventional order: truncate the tail, then temperature, then draw.
        if (sp.top_k > 0) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp.top_k));
        }
        if (sp.top_p < 1.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp.top_p, 1));
        }
        llama_sampler_chain_add(chain, llama_sampler_init_temp(sp.temp));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(sp.seed));
    }
    return chain;
}

void Context::close() {
    for (auto & kv : seq_smpl_) {
        if (kv.second != nullptr) {
            llama_sampler_free(kv.second);
        }
    }
    seq_smpl_.clear();
    if (smpl_ != nullptr) {
        llama_sampler_free(smpl_);
        smpl_ = nullptr;
    }
    if (ctx_ != nullptr) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    last_logits_.clear();
    expert_frames_.clear();
    expert_frames_.shrink_to_fit();
}

Context::~Context() {
    close();
}

void Context::ensure_open() const {
    if (ctx_ == nullptr) {
        throw std::runtime_error(
            "this Context has been closed; create a new one to keep serving");
    }
}

void Context::decode_seq0(const std::vector<llama_token> & tokens) {
    ensure_open();
    if (tokens.empty()) {
        return;
    }
    // Hard guard, not an optimization: llama_decode asserts n_tokens <= n_batch, and a
    // failed GGML_ASSERT calls abort() -- it does not throw. In the single-process server
    // design an oversized batch would take down the API server along with the engine, so
    // every batch is bounds-checked on this side of the call and raised as a Python
    // exception instead. Callers chunk their prefill to n_batch (see generate.py).
    const uint32_t n_batch = llama_n_batch(ctx_);
    if (tokens.size() > n_batch) {
        throw std::invalid_argument(
            "batch of " + std::to_string(tokens.size()) + " tokens exceeds n_batch=" +
            std::to_string(n_batch) + "; split the prefill into n_batch-sized chunks");
    }
    // llama_batch_get_one takes a non-const pointer but does not write through it.
    auto * data = const_cast<llama_token *>(tokens.data());
    llama_batch batch = llama_batch_get_one(data, (int32_t) tokens.size());

    decode_raw(batch, (int32_t) tokens.size());
}

void Context::decode(const Batch & batch) {
    ensure_open();
    const int32_t n_tokens = batch.n_tokens();
    if (n_tokens == 0) {
        // llama_decode rejects an empty batch with -1; nothing to do is not an error.
        return;
    }

    // Same discipline as decode_seq0: every constraint llama.cpp enforces with a
    // GGML_ASSERT (which abort()s) is checked here and raised instead.
    const uint32_t nb = llama_n_batch(ctx_);
    if ((uint32_t) n_tokens > nb) {
        throw std::invalid_argument(
            "batch of " + std::to_string(n_tokens) + " tokens exceeds n_batch=" +
            std::to_string(nb) + "; split the step into n_batch-sized chunks");
    }
    const uint32_t ns = llama_n_seq_max(ctx_);
    if (batch.max_seq_id() >= (llama_seq_id) ns) {
        throw std::invalid_argument(
            "batch references seq_id " + std::to_string(batch.max_seq_id()) +
            " but the context was created with n_seq_max=" + std::to_string(ns));
    }
    const uint32_t nc = llama_n_ctx(ctx_);
    if (batch.max_pos() >= (llama_pos) nc) {
        throw std::invalid_argument(
            "batch references position " + std::to_string(batch.max_pos()) +
            " but the context holds n_ctx=" + std::to_string(nc) + " tokens");
    }

    decode_raw(batch.raw(), n_tokens);
}

void Context::decode_raw(const llama_batch & batch, int32_t n_tokens) {
    // The single llama_decode() call site. Counted here rather than in Python so the
    // "one decode per step" invariant is measured where it actually happens.
    // Recorded router frames belong to exactly one decode the same way: a new
    // decode starts a new frame set, so a reader always sees one decode.
    expert_frames_.clear();
    const int32_t rc = llama_decode(ctx_, batch);
    ++decode_calls_;

    // Handle failure BEFORE recording the logits mask, and clear the mask on the way
    // out. Recording first is unsound in two distinct ways:
    //
    //  1. llama.cpp returns rc=1 (no KV slot) before it reaches output_reserve(), so
    //     `n_outputs` keeps its previous value while the mask would already claim a
    //     flagged row. any_row_has_logits() would then pass and get_logits_ith() would
    //     be the only thing between the caller and llama-sampler.cpp's
    //     GGML_ASSERT(logits != nullptr) -- and that nullptr return exists only under
    //     NDEBUG. In a Debug build the assert fires and abort()s the process.
    //  2. After a failed decode that FOLLOWS a successful one, `n_outputs` survives, so
    //     sampling would silently return the PREVIOUS decode's logits -- a wrong token
    //     reported as a good one, which is worse than an error.
    //
    // Clearing makes both cases raise from the first guard: no logits are claimed for a
    // batch that did not decode.
    if (rc != 0) {
        last_logits_.clear();
        if (rc == 1) {
            throw std::runtime_error("llama_decode: no KV slot for batch (raise n_ctx or shrink the batch)");
        }
        if (rc == 2) {
            throw std::runtime_error("llama_decode was aborted");
        }
        throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));
    }

    // Record the logits mask of what we just decoded so sample_seq() can validate its
    // row index. llama_batch_get_one leaves .logits null, which means "last token only".
    last_logits_.assign((size_t) n_tokens, 0);
    if (batch.logits != nullptr) {
        for (int32_t i = 0; i < n_tokens; ++i) {
            last_logits_[(size_t) i] = batch.logits[i];
        }
    } else if (n_tokens > 0) {
        last_logits_[(size_t) n_tokens - 1] = 1;
    }
}

llama_token Context::sample_last() {
    ensure_open();
    // Same two-stage guard as sample_seq(), against the same abort. Index -1 means "the
    // last OUTPUT row", so it is only meaningful when the last decoded batch produced at
    // least one logits row. Phase 0's decode_seq0 goes through llama_batch_get_one, whose
    // null .logits means "last token only", so decode_raw always records a set flag there
    // and that path is unaffected. Phase 1's decode(const Batch &) is the first way to
    // leave the context with NO logits at all -- MetalEngine does exactly that on every
    // intermediate chunk of a chunked prefill -- and llama_sampler_sample() would then
    // hit llama-sampler.cpp's GGML_ASSERT(logits != nullptr) and abort() the process
    // (killing the API server) instead of throwing.
    if (!any_row_has_logits()) {
        throw std::invalid_argument(
            "sample_last(): the last decoded batch produced no logits (" +
            std::to_string(last_logits_.size()) +
            " tokens decoded, none flagged for output); decode a batch that requests "
            "logits, or sample a flagged row with sample_seq(seq_id, idx)");
    }
    // Second belt, as in sample_seq(): llama_get_logits_ith() is what
    // llama_sampler_sample() reads and it is the thing that asserts. Ask for the row
    // first -- it returns null rather than aborting -- and raise that as an exception.
    if (llama_get_logits_ith(ctx_, -1) == nullptr) {
        throw std::runtime_error(
            "sample_last(): no logits available for the last decoded batch");
    }
    return llama_sampler_sample(smpl_, ctx_, -1);
}

void Context::accept(llama_token tok) {
    ensure_open();
    llama_sampler_accept(smpl_, tok);
}

void Context::validate_seq_id(llama_seq_id seq_id) const {
    ensure_open();
    const uint32_t ns = llama_n_seq_max(ctx_);
    if (seq_id < 0 || (uint32_t) seq_id >= ns) {
        throw std::invalid_argument("seq_id " + std::to_string(seq_id) +
                                    " out of range [0, " + std::to_string(ns) + ")");
    }
}

void Context::set_seq_sampler(llama_seq_id seq_id, const SamplerParams & sp) {
    ensure_open();
    validate_seq_id(seq_id);
    llama_sampler * chain = make_chain(sp);  // may throw; nothing installed yet
    auto it = seq_smpl_.find(seq_id);
    if (it != seq_smpl_.end()) {
        llama_sampler_free(it->second);
        it->second = chain;
    } else {
        seq_smpl_.emplace(seq_id, chain);
    }
}

void Context::reset_seq_sampler(llama_seq_id seq_id) {
    auto it = seq_smpl_.find(seq_id);
    if (it != seq_smpl_.end()) {
        llama_sampler_free(it->second);
        seq_smpl_.erase(it);
    }
}

bool Context::has_seq_sampler(llama_seq_id seq_id) const {
    return seq_smpl_.find(seq_id) != seq_smpl_.end();
}

llama_sampler * Context::seq_sampler(llama_seq_id seq_id) const {
    auto it = seq_smpl_.find(seq_id);
    if (it == seq_smpl_.end()) {
        throw std::invalid_argument("no sampler installed for seq_id " + std::to_string(seq_id) +
                                    "; call set_seq_sampler first");
    }
    return it->second;
}

bool Context::row_has_logits(int32_t idx) const {
    return idx >= 0 && (size_t) idx < last_logits_.size() && last_logits_[(size_t) idx] != 0;
}

bool Context::any_row_has_logits() const {
    for (int8_t flag : last_logits_) {
        if (flag != 0) {
            return true;
        }
    }
    return false;
}

llama_token Context::sample_seq(llama_seq_id seq_id, int32_t idx) {
    ensure_open();
    validate_seq_id(seq_id);
    llama_sampler * chain = seq_sampler(seq_id);

    if (!row_has_logits(idx)) {
        throw std::invalid_argument(
            "batch row " + std::to_string(idx) + " produced no logits (last batch had " +
            std::to_string(last_logits_.size()) + " tokens); set its logits flag to sample it");
    }
    // Second belt: get_logits_ith() is what llama_sampler_sample() reads, and it
    // GGML_ASSERTs on a null result. Ask for the row first and fail as an exception.
    if (llama_get_logits_ith(ctx_, idx) == nullptr) {
        throw std::runtime_error("no logits available for batch row " + std::to_string(idx));
    }
    return llama_sampler_sample(chain, ctx_, idx);
}

void Context::accept_seq(llama_seq_id seq_id, llama_token tok) {
    ensure_open();
    validate_seq_id(seq_id);
    llama_sampler_accept(seq_sampler(seq_id), tok);
}

bool Context::memory_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    ensure_open();
    // llama.h documents `seq_id < 0` as "match any sequence", but the implementation
    // exempts exactly -1: llama_kv_cache::seq_rm asserts
    // `seq_id == -1 || (seq_id >= 0 && seq_id < seq_to_stream.size())`, and a failed
    // GGML_ASSERT abort()s the process. So -1 passes through as the documented
    // wildcard and every other value must be a real, in-range id.
    if (seq_id != -1) {
        validate_seq_id(seq_id);
    }
    // p0/p1 need no guard here: seq_rm clamps negatives to [0, inf) and simply matches
    // no cells for a range past the end. The bool is load-bearing (see the header):
    // pass it through so a caller that packed past the rewind point can refuse to
    // proceed on false instead of desyncing into the next decode.
    return llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, p0, p1);
}

void Context::memory_seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
    ensure_open();
    validate_seq_id(src);
    validate_seq_id(dst);

    // The positions need a guard of their own. With kv_unified == false (llama.cpp's
    // default) every sequence gets its own KV stream, so a copy between two different
    // sequences is a cross-stream copy of real buffer data -- which llama.cpp only
    // implements for the entire buffer:
    //   llama-kv-cache.cpp: GGML_ASSERT(is_full && "seq_cp() is only supported for
    //                                    full KV buffers")
    // and that assert abort()s. A unified buffer puts both sequences in stream 0,
    // where seq_cp is pure cell metadata and any sub-range is legal; that partial
    // copy (fork the prefix [0, anchor)) is the primitive prefix reuse needs, so the
    // guard names the flag instead of silently widening the caller's range.
    // A negative or zero bound is llama.cpp's own "full buffer" spelling, and a copy
    // onto itself returns before the assert, so both stay allowed.
    const bool full_range = p0 <= 0 && p1 <= 0;
    if (src != dst && !full_range && !kv_unified_) {
        throw std::invalid_argument(
            "memory_seq_cp of the partial range [" + std::to_string(p0) + ", " +
            std::to_string(p1) + ") from seq_id " + std::to_string(src) + " to seq_id " +
            std::to_string(dst) + " needs a unified KV buffer: create the context with "
            "kv_unified=true, or copy the full sequence with p0=-1, p1=-1");
    }
    llama_memory_seq_cp(llama_get_memory(ctx_), src, dst, p0, p1);
}

void Context::memory_seq_keep(llama_seq_id seq_id) {
    ensure_open();
    validate_seq_id(seq_id);
    llama_memory_seq_keep(llama_get_memory(ctx_), seq_id);
}

// Geometry accessors read through ctx_, so they must refuse a closed context too --
// a health endpoint reading n_ctx during shutdown would otherwise null-deref.
uint32_t Context::n_ctx()     const { ensure_open(); return llama_n_ctx(ctx_);     }
uint32_t Context::n_batch()   const { ensure_open(); return llama_n_batch(ctx_);   }
uint32_t Context::n_ubatch()  const { ensure_open(); return llama_n_ubatch(ctx_);  }
uint32_t Context::n_seq_max() const { ensure_open(); return llama_n_seq_max(ctx_); }
uint32_t Context::n_ctx_seq() const { ensure_open(); return llama_n_ctx_seq(ctx_); }
uint32_t Context::n_rs_seq()  const { ensure_open(); return llama_n_rs_seq(ctx_);  }

bool Context::expert_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    // Router distributions are named "ffn_moe_probs-{layer}" by llama.cpp's
    // graph builder for every MoE arch -- match that EXACTLY (trailing layer
    // digits only). Sibling nodes share the prefix ("ffn_moe_probs_biased",
    // "ffn_moe_probs_masked") but are selection intermediates with different
    // shapes, not distributions; matching them would corrupt every consumer.
    static const char kPrefix[] = "ffn_moe_probs-";
    static const size_t kLen = sizeof(kPrefix) - 1;
    if (t == nullptr || strncmp(t->name, kPrefix, kLen) != 0) {
        return false;
    }
    // The suffix must be the bare layer index: "ffn_moe_probs_biased-N" and
    // "ffn_moe_probs_masked-N" are selection intermediates with different
    // shapes, and capturing them would corrupt every consumer downstream.
    for (const char * p = t->name + kLen; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            return false;
        }
    }
    if (t->name[kLen] == '\0') {
        return false;  // prefix with no layer: not a real node, stay fused
    }
    if (ask) {
        return true;
    }
    // ask=false runs post-compute under the scheduler's own synchronization,
    // so t->data is valid to copy (even on Metal's shared buffers). Returning
    // false HERE would abort the whole remaining compute loop, so every path
    // below returns true: a useless frame is harmless, a truncated decode is
    // not. Only F32 router outputs are captured; anything else is skipped.
    if (t->type != GGML_TYPE_F32) {
        return true;
    }
    const int64_t n_tokens = t->ne[1];
    const int64_t n_expert = t->ne[0];
    if (n_tokens <= 0 || n_expert <= 0 || t->data == nullptr) {
        return true;
    }
    auto * self = static_cast<Context *>(user_data);
    ExpertFrame frame;
    frame.layer = std::atoi(t->name + kLen);
    frame.n_tokens = n_tokens;
    const float * data = static_cast<const float *>(t->data);
    frame.probs.assign(data, data + n_tokens * n_expert);
    self->expert_frames_.push_back(std::move(frame));
    return true;
}

std::vector<ExpertFrame> Context::expert_activations() {
    ensure_open();
    std::vector<ExpertFrame> out;
    out.swap(expert_frames_);
    return out;
}

} // namespace ftm
