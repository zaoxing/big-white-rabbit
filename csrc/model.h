// Model handle: owns a llama_model and exposes vocab/metadata queries.
//
// Phase 0 scope: load + tokenize + metadata. Expert-tensor placement
// (llama_model_params.tensor_buft_overrides) lands in buffer_control.* in Phase 3;
// the ModelParams struct below already carries the field so the policy layer has a
// place to write into without reshaping this interface.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "llama.h"

namespace ftm {

// Initializes the ggml/llama backend once per process. Safe to call repeatedly.
void backend_init_once();

struct ModelParams {
    // -1 means "all layers on the GPU". On unified memory this is the sane default:
    // there is no separate VRAM pool to run out of, so partial offload only costs
    // performance. Phase 3's policy narrows this per-tensor instead of per-layer.
    int32_t n_gpu_layers = -1;
    // llama_load_mode: AUTO(-1) picks per device capability. Replaces the old
    // use_mmap/use_mlock booleans, and adds DIRECT_IO.
    llama_load_mode load_mode = LLAMA_LOAD_MODE_AUTO;
    // llama_lazy_mode: on-demand row reads for arch-marked tensors (requires mmap).
    // Phase 3 should revisit this -- it is a second, finer-grained residency knob than
    // tensor_buft_overrides, and interacts directly with the RAM-budget term.
    llama_lazy_mode lazy_mode = LLAMA_LAZY_MODE_AUTO;
    // Metadata-only load with simulated allocations -- lets the placement solver size a
    // model without paying for it. (llama_model_params.no_alloc)
    bool    no_alloc     = false;
};

class Model {
public:
    Model(const std::string & path, const ModelParams & params);
    ~Model();

    Model(const Model &)             = delete;
    Model & operator=(const Model &) = delete;

    llama_model       * raw()   const { return model_; }
    const llama_vocab * vocab() const { return vocab_; }
    const std::string & path()  const { return path_;  }

    std::vector<llama_token> tokenize(const std::string & text, bool add_special, bool parse_special) const;
    std::string token_to_piece(llama_token tok, bool special) const;
    std::string detokenize(const std::vector<llama_token> & toks, bool unparse_special) const;
    bool is_eog(llama_token tok) const;

    int32_t  n_ctx_train() const;
    int32_t  n_embd()      const;
    int32_t  n_layer()     const;
    int32_t  n_vocab()     const;
    uint64_t size_bytes()  const;
    uint64_t n_params()    const;
    std::string desc()     const;

    // GGUF KV lookup by key (e.g. "general.architecture"). Empty string when absent.
    std::string meta_val(const std::string & key) const;

private:
    std::string         path_;
    llama_model       * model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
};

} // namespace ftm
