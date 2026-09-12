#include "model.h"

#include <mutex>
#include <stdexcept>

#include <ggml-backend.h>

namespace bwr {

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

    if (params.expert_weights == "cpu") {
        // Same regex as upstream --cpu-moe (common/common.h LLM_FFN_EXPS_REGEX):
        // every MoE expert tensor, std::regex substring-matched by the loader.
        // select_weight_buft routes CPU-overridden tensors through the CPU
        // buffer list (extra bufts / repacking as applicable).
        override_patterns_.emplace_back(R"(\.ffn_(up|down|gate|gate_up)_(ch|)exps)");
        overrides_.push_back(
            {override_patterns_.back().c_str(), ggml_backend_cpu_buffer_type()});
        overrides_.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = overrides_.data();
    } else if (params.expert_weights != "metal") {
        throw std::invalid_argument(
            "ModelParams.expert_weights must be \"metal\" or \"cpu\"; got \"" +
            params.expert_weights + "\"");
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

void Model::close() {
    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
        vocab_ = nullptr;  // borrowed from the model; invalid once it is freed
    }
}

Model::~Model() {
    close();
}

void Model::ensure_open() const {
    if (model_ == nullptr) {
        throw std::runtime_error("this Model has been closed; reload it to keep serving");
    }
}

std::vector<llama_token> Model::tokenize(const std::string & text, bool add_special, bool parse_special) const {
    ensure_open();
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
    ensure_open();
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
    ensure_open();
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

namespace {

// llama.cpp's string getters are snprintf-backed, so they return the length the value
// WOULD have needed -- not the number of bytes written. Two bugs follow from treating
// that as a written-length: `std::string(buf, n)` with n > sizeof(buf) reads PAST the
// buffer (a stack over-read, and whatever it picks up ends up in a Python string), and
// silently clamping to the buffer edge can split a multi-byte UTF-8 sequence, which then
// fails to decode. GGUF chat templates run to several KiB, so this is routine, not
// theoretical. Probe for the real length, then allocate exactly once.
template <typename Fill>
std::string read_sized_string(Fill fill) {
    char probe[256];
    const int32_t n = fill(probe, sizeof(probe));
    if (n < 0) {
        return std::string();  // absent / not applicable
    }
    if ((size_t) n < sizeof(probe)) {
        return std::string(probe, (size_t) n);
    }
    // +1: the getters write a null terminator inside buf_size (see llama.h).
    std::string out((size_t) n + 1, '\0');
    const int32_t n2 = fill(out.data(), out.size());
    if (n2 < 0) {
        return std::string();
    }
    out.resize((size_t) (n2 < n ? n2 : n));
    return out;
}

} // namespace

std::string Model::desc() const {
    return read_sized_string([this](char * buf, size_t len) {
        return llama_model_desc(model_, buf, len);
    });
}

std::string Model::meta_val(const std::string & key) const {
    return read_sized_string([this, &key](char * buf, size_t len) {
        return llama_model_meta_val_str(model_, key.c_str(), buf, len);
    });
}

std::string Model::chat_template() const {
    ensure_open();
    // The GGUF-embedded template (llama_model_chat_template returns a borrowed pointer
    // into the model, or nullptr when the model carries none).
    const char * tmpl = llama_model_chat_template(model_, nullptr);
    return tmpl != nullptr ? std::string(tmpl) : std::string();
}

std::string Model::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>> & messages,
    bool add_assistant) const {
    ensure_open();
    if (messages.empty()) {
        throw std::invalid_argument("apply_chat_template: no messages");
    }
    const std::string tmpl = chat_template();
    if (tmpl.empty()) {
        throw std::runtime_error(
            "model carries no chat template (GGUF key tokenizer.chat_template); "
            "send a raw prompt instead of messages");
    }

    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    for (const auto & m : messages) {
        chat.push_back({m.first.c_str(), m.second.c_str()});
    }

    // Same snprintf-style sizing contract as above, but llama_chat_apply_template
    // returns a NEGATIVE value for "template not recognized" rather than a length --
    // it pattern-matches against a fixed list of ~56 known templates and does NOT
    // evaluate jinja (llama.h:1215). So a negative result here is a real failure, not
    // a resize hint, and it must be reported rather than retried.
    int32_t n = llama_chat_apply_template(tmpl.c_str(), chat.data(), chat.size(),
                                          add_assistant, nullptr, 0);
    if (n < 0) {
        throw std::runtime_error(
            "this model's chat template is not one llama.cpp can apply (it matches "
            "templates by pattern and does not evaluate jinja); send a raw prompt "
            "instead of messages");
    }
    std::string out((size_t) n + 1, '\0');
    n = llama_chat_apply_template(tmpl.c_str(), chat.data(), chat.size(),
                                 add_assistant, out.data(), (int32_t) out.size());
    if (n < 0) {
        throw std::runtime_error("chat template application failed on the second pass");
    }
    out.resize((size_t) n);
    return out;
}

} // namespace bwr
