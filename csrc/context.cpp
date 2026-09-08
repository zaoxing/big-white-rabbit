#include "context.h"

#include <stdexcept>
#include <string>

namespace ftm {

Context::Context(std::shared_ptr<Model> model, const ContextParams & cp, const SamplerParams & sp)
    : model_(std::move(model)) {
    if (!model_) {
        throw std::invalid_argument("Context requires a loaded Model");
    }

    llama_context_params lp = llama_context_default_params();
    lp.n_ctx           = cp.n_ctx;
    lp.n_batch         = cp.n_batch;
    lp.n_ubatch        = cp.n_ubatch;
    lp.n_seq_max       = cp.n_seq_max;
    lp.flash_attn_type = cp.flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO
                                       : LLAMA_FLASH_ATTN_TYPE_DISABLED;
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

    llama_sampler_chain_params scp = llama_sampler_chain_default_params();
    smpl_ = llama_sampler_chain_init(scp);
    if (smpl_ == nullptr) {
        llama_free(ctx_);
        ctx_ = nullptr;
        throw std::runtime_error("failed to create sampler chain");
    }

    if (sp.temp <= 0.0f) {
        llama_sampler_chain_add(smpl_, llama_sampler_init_greedy());
    } else {
        // Conventional order: truncate the tail, then temperature, then draw.
        if (sp.top_k > 0) {
            llama_sampler_chain_add(smpl_, llama_sampler_init_top_k(sp.top_k));
        }
        if (sp.top_p < 1.0f) {
            llama_sampler_chain_add(smpl_, llama_sampler_init_top_p(sp.top_p, 1));
        }
        llama_sampler_chain_add(smpl_, llama_sampler_init_temp(sp.temp));
        llama_sampler_chain_add(smpl_, llama_sampler_init_dist(sp.seed));
    }
}

Context::~Context() {
    if (smpl_ != nullptr) {
        llama_sampler_free(smpl_);
        smpl_ = nullptr;
    }
    if (ctx_ != nullptr) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
}

void Context::decode_seq0(const std::vector<llama_token> & tokens) {
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

    const int32_t rc = llama_decode(ctx_, batch);
    if (rc == 1) {
        throw std::runtime_error("llama_decode: no KV slot for batch (raise n_ctx or shrink the batch)");
    }
    if (rc < 0) {
        throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));
    }
}

llama_token Context::sample_last() {
    return llama_sampler_sample(smpl_, ctx_, -1);
}

void Context::accept(llama_token tok) {
    llama_sampler_accept(smpl_, tok);
}

void Context::memory_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, p0, p1);
}

uint32_t Context::n_ctx()    const { return llama_n_ctx(ctx_);    }
uint32_t Context::n_batch()  const { return llama_n_batch(ctx_);  }
uint32_t Context::n_ubatch() const { return llama_n_ubatch(ctx_); }

} // namespace ftm
