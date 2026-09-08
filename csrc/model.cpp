#include "model.h"

#include <mutex>
#include <stdexcept>

namespace ftm {

void backend_init_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_backend_init();
        // No llama_backend_free() here: teardown order between Python's atexit and
        // Metal's device teardown is not worth racing over for a process about to exit.
    });
}

Model::Model(const std::string & path, const ModelParams & params) : path_(path) {
    backend_init_once();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = params.n_gpu_layers;
    mp.load_mode    = params.load_mode;
    mp.lazy_mode    = params.lazy_mode;
    mp.no_alloc     = params.no_alloc;

    if (params.no_alloc) {
        // no_alloc is llama.cpp's "memory-fit pass": it maps nothing and only simulates
        // allocations, which is exactly what a placement/sizing solver wants. But the
        // mmap path asserts !no_alloc (llama-model.cpp: GGML_ASSERT(!ml.no_alloc)), and
        // LOAD_MODE_AUTO resolves to mmap on Apple Silicon -- so a sizing load must opt
        // out of mmap and lazy mapping rather than inherit the caller's defaults.
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    }

    model_ = llama_model_load_from_file(path.c_str(), mp);
    if (model_ == nullptr) {
        throw std::runtime_error("failed to load model: " + path);
    }

    vocab_ = llama_model_get_vocab(model_);
    if (vocab_ == nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("model has no vocab: " + path);
    }
}

Model::~Model() {
    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

std::vector<llama_token> Model::tokenize(const std::string & text, bool add_special, bool parse_special) const {
    // Negative return = -(required capacity); retry once at that size.
    int32_t n_upper = -llama_tokenize(vocab_, text.data(), (int32_t) text.size(),
                                      nullptr, 0, add_special, parse_special);
    std::vector<llama_token> out(n_upper > 0 ? n_upper : 0);
    const int32_t n = llama_tokenize(vocab_, text.data(), (int32_t) text.size(),
                                     out.data(), (int32_t) out.size(), add_special, parse_special);
    if (n < 0) {
        throw std::runtime_error("tokenize failed: buffer too small after resize");
    }
    out.resize(n);
    return out;
}

std::string Model::token_to_piece(llama_token tok, bool special) const {
    char buf[256];
    const int32_t n = llama_token_to_piece(vocab_, tok, buf, sizeof(buf), 0, special);
    if (n < 0) {
        std::string big(-n, '\0');
        const int32_t n2 = llama_token_to_piece(vocab_, tok, big.data(), (int32_t) big.size(), 0, special);
        if (n2 < 0) {
            throw std::runtime_error("token_to_piece failed for token " + std::to_string(tok));
        }
        big.resize(n2);
        return big;
    }
    return std::string(buf, n);
}

std::string Model::detokenize(const std::vector<llama_token> & toks, bool unparse_special) const {
    std::string out(toks.size() * 4 + 16, '\0');
    int32_t n = llama_detokenize(vocab_, toks.data(), (int32_t) toks.size(),
                                 out.data(), (int32_t) out.size(), false, unparse_special);
    if (n < 0) {
        out.resize(-n);
        n = llama_detokenize(vocab_, toks.data(), (int32_t) toks.size(),
                             out.data(), (int32_t) out.size(), false, unparse_special);
        if (n < 0) {
            throw std::runtime_error("detokenize failed");
        }
    }
    out.resize(n);
    return out;
}

bool     Model::is_eog(llama_token tok) const { return llama_vocab_is_eog(vocab_, tok); }
int32_t  Model::n_ctx_train()           const { return llama_model_n_ctx_train(model_); }
int32_t  Model::n_embd()                const { return llama_model_n_embd(model_); }
int32_t  Model::n_layer()               const { return llama_model_n_layer(model_); }
int32_t  Model::n_vocab()               const { return llama_vocab_n_tokens(vocab_); }
uint64_t Model::size_bytes()            const { return llama_model_size(model_); }
uint64_t Model::n_params()              const { return llama_model_n_params(model_); }

std::string Model::desc() const {
    char buf[512];
    const int32_t n = llama_model_desc(model_, buf, sizeof(buf));
    return n > 0 ? std::string(buf, n) : std::string();
}

std::string Model::meta_val(const std::string & key) const {
    char buf[1024];
    const int32_t n = llama_model_meta_val_str(model_, key.c_str(), buf, sizeof(buf));
    return n > 0 ? std::string(buf, n) : std::string();
}

} // namespace ftm
