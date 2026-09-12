// pybind11 entry point for the Big White Rabbit Metal engine.
//
// Every call that can block for a meaningful time (model load, decode) releases the GIL.
// That is load-bearing for the single-process control-plane design: uvicorn's event loop
// and the decode loop share this process, so a decode that held the GIL would stall
// /health and every concurrent request.

#include <memory>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "batch.h"
#include "context.h"
#include "model.h"

namespace py = pybind11;

PYBIND11_MODULE(_bwr_metal, m) {
    // pybind11's default translators already map std::invalid_argument and
    // std::length_error (Batch's overflow guard) onto Python ValueError, so every
    // bounds check added in Phase 1 reaches Python as an exception, not an abort().
    m.doc() = "Big White Rabbit: llama.cpp/Metal bindings (Phase 1)";

    m.def("backend_init", &bwr::backend_init_once,
          "Initialize the ggml/llama backend (idempotent; called on module import).");

    py::enum_<llama_load_mode>(m, "LoadMode")
        .value("AUTO",       LLAMA_LOAD_MODE_AUTO)
        .value("NONE",       LLAMA_LOAD_MODE_NONE)
        .value("MMAP",       LLAMA_LOAD_MODE_MMAP)
        .value("MLOCK",      LLAMA_LOAD_MODE_MLOCK)
        .value("MMAP_MLOCK", LLAMA_LOAD_MODE_MMAP_MLOCK)
        .value("DIRECT_IO",  LLAMA_LOAD_MODE_DIRECT_IO);

    py::enum_<llama_lazy_mode>(m, "LazyMode")
        .value("OFF",  LLAMA_LAZY_MODE_OFF)
        .value("AUTO", LLAMA_LAZY_MODE_AUTO)
        .value("ON",   LLAMA_LAZY_MODE_ON);

    py::class_<bwr::ModelParams>(m, "ModelParams")
        .def(py::init<>())
        .def_readwrite("n_gpu_layers", &bwr::ModelParams::n_gpu_layers)
        .def_readwrite("load_mode",    &bwr::ModelParams::load_mode)
        .def_readwrite("lazy_mode",    &bwr::ModelParams::lazy_mode)
        .def_readwrite("no_alloc",     &bwr::ModelParams::no_alloc)
        // MoE expert residency: "metal" (default) or "cpu" (SPEC-expert-placement.md).
        .def_readwrite("expert_weights", &bwr::ModelParams::expert_weights);

    py::class_<bwr::ContextParams>(m, "ContextParams")
        .def(py::init<>())
        .def_readwrite("n_ctx",           &bwr::ContextParams::n_ctx)
        .def_readwrite("n_batch",         &bwr::ContextParams::n_batch)
        .def_readwrite("n_ubatch",        &bwr::ContextParams::n_ubatch)
        .def_readwrite("n_seq_max",       &bwr::ContextParams::n_seq_max)
        .def_readwrite("n_threads",       &bwr::ContextParams::n_threads)
        .def_readwrite("n_threads_batch", &bwr::ContextParams::n_threads_batch)
        .def_readwrite("flash_attn",      &bwr::ContextParams::flash_attn)
        // Unified KV buffer: required for a partial-range memory_seq_cp (prefix fork).
        .def_readwrite("kv_unified",      &bwr::ContextParams::kv_unified)
        // Recurrent-state rollback snapshots for partial memory_seq_rm on hybrids.
        .def_readwrite("n_rs_seq",        &bwr::ContextParams::n_rs_seq)
        // MoE router distribution recording (one graph split per MoE layer
        // per decode while on -- profile, don't serve, with it).
        .def_readwrite("record_experts",  &bwr::ContextParams::record_experts);

    py::class_<bwr::ExpertFrame>(m, "ExpertFrame")
        .def_readonly("layer", &bwr::ExpertFrame::layer)
        .def_readonly("n_tokens", &bwr::ExpertFrame::n_tokens)
        .def_readonly("probs", &bwr::ExpertFrame::probs);

    py::class_<bwr::SamplerParams>(m, "SamplerParams")
        .def(py::init<>())
        .def_readwrite("temp",  &bwr::SamplerParams::temp)
        .def_readwrite("top_k", &bwr::SamplerParams::top_k)
        .def_readwrite("top_p", &bwr::SamplerParams::top_p)
        .def_readwrite("seed",  &bwr::SamplerParams::seed);

    py::class_<bwr::Model, std::shared_ptr<bwr::Model>>(m, "Model")
        .def(py::init<const std::string &, const bwr::ModelParams &>(),
             py::arg("path"), py::arg("params") = bwr::ModelParams(),
             py::call_guard<py::gil_scoped_release>())
        .def("tokenize", &bwr::Model::tokenize,
             py::arg("text"), py::arg("add_special") = true, py::arg("parse_special") = true)
        .def("token_to_piece", &bwr::Model::token_to_piece,
             py::arg("token"), py::arg("special") = false)
        .def("detokenize", &bwr::Model::detokenize,
             py::arg("tokens"), py::arg("unparse_special") = false)
        .def("is_eog", &bwr::Model::is_eog, py::arg("token"))
        .def("meta_val", &bwr::Model::meta_val, py::arg("key"))
        .def("apply_chat_template", &bwr::Model::apply_chat_template,
             py::arg("messages"), py::arg("add_assistant") = true,
             "Render [(role, content), ...] into a prompt via the model's own template.")
        .def_property_readonly("chat_template", &bwr::Model::chat_template)
        .def_property_readonly("path",        &bwr::Model::path)
        .def_property_readonly("n_ctx_train", &bwr::Model::n_ctx_train)
        .def_property_readonly("n_embd",      &bwr::Model::n_embd)
        .def_property_readonly("n_layer",     &bwr::Model::n_layer)
        .def_property_readonly("n_vocab",     &bwr::Model::n_vocab)
        .def_property_readonly("size_bytes",  &bwr::Model::size_bytes)
        .def_property_readonly("n_params",    &bwr::Model::n_params)
        .def_property_readonly("desc",        &bwr::Model::desc)
        .def("close", &bwr::Model::close,
             "Free the weights now. Close every Context on this model FIRST; required "
             "for deterministic shutdown (see Model::close in model.h).")
        .def_property_readonly("closed", &bwr::Model::closed)
        .def("__repr__", [](const bwr::Model & self) {
            return self.closed() ? std::string("<bwr.Model closed>")
                                 : "<bwr.Model '" + self.desc() + "'>";
        });

    py::class_<bwr::Batch>(m, "Batch")
        .def(py::init<int32_t, int32_t>(),
             py::arg("capacity"), py::arg("n_seq_max_per_token") = 1)
        .def("clear", &bwr::Batch::clear)
        .def("add", &bwr::Batch::add,
             py::arg("token"), py::arg("pos"), py::arg("seq_id"), py::arg("logits") = false,
             "Append a token; returns its batch row index (raises past capacity).")
        .def("add_shared", &bwr::Batch::add_shared,
             py::arg("token"), py::arg("pos"), py::arg("seq_ids"), py::arg("logits") = false)
        .def_property_readonly("n_tokens", &bwr::Batch::n_tokens)
        .def_property_readonly("capacity", &bwr::Batch::capacity)
        .def_property_readonly("n_seq_max_per_token", &bwr::Batch::n_seq_max_per_token)
        .def_property_readonly("max_seq_id", &bwr::Batch::max_seq_id)
        .def_property_readonly("max_pos", &bwr::Batch::max_pos)
        .def("__len__", &bwr::Batch::n_tokens)
        .def("__repr__", [](const bwr::Batch & self) {
            return "<bwr.Batch " + std::to_string(self.n_tokens()) + "/" +
                   std::to_string(self.capacity()) + " tokens>";
        });

    py::class_<bwr::Context>(m, "Context")
        .def(py::init<std::shared_ptr<bwr::Model>, const bwr::ContextParams &, const bwr::SamplerParams &>(),
             py::arg("model"),
             py::arg("params")  = bwr::ContextParams(),
             py::arg("sampler") = bwr::SamplerParams(),
             py::call_guard<py::gil_scoped_release>())
        .def("decode_seq0", &bwr::Context::decode_seq0, py::arg("tokens"),
             py::call_guard<py::gil_scoped_release>())
        .def("sample_last", &bwr::Context::sample_last,
             py::call_guard<py::gil_scoped_release>())
        .def("accept", &bwr::Context::accept, py::arg("token"))
        // Batched step. The caller's reference keeps `batch` alive for the whole call,
        // so releasing the GIL cannot let Python free the arrays llama_decode reads.
        .def("decode", &bwr::Context::decode, py::arg("batch"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_seq_sampler", &bwr::Context::set_seq_sampler,
             py::arg("seq_id"), py::arg("sampler") = bwr::SamplerParams())
        .def("reset_seq_sampler", &bwr::Context::reset_seq_sampler, py::arg("seq_id"))
        .def("has_seq_sampler", &bwr::Context::has_seq_sampler, py::arg("seq_id"))
        .def("sample_seq", &bwr::Context::sample_seq, py::arg("seq_id"), py::arg("idx"),
             py::call_guard<py::gil_scoped_release>())
        .def("accept_seq", &bwr::Context::accept_seq, py::arg("seq_id"), py::arg("token"))
        .def("row_has_logits", &bwr::Context::row_has_logits, py::arg("idx"))
        // sample_last()'s precondition, exposed so a caller can ask instead of catching.
        .def_property_readonly("any_row_has_logits", &bwr::Context::any_row_has_logits)
        .def("memory_seq_rm", &bwr::Context::memory_seq_rm,
             py::arg("seq_id"), py::arg("p0") = -1, py::arg("p1") = -1,
             "Drop a KV range; returns llama.cpp's verdict. A partial rm can "
             "report False (hybrid without rollback snapshots) -- callers that "
             "packed past the rewind point must check.")
        .def("memory_seq_cp", &bwr::Context::memory_seq_cp,
             py::arg("src"), py::arg("dst"), py::arg("p0") = -1, py::arg("p1") = -1)
        .def("memory_seq_keep", &bwr::Context::memory_seq_keep, py::arg("seq_id"))
        .def("state_seq_get_size", &bwr::Context::state_seq_get_size, py::arg("seq_id"),
             py::call_guard<py::gil_scoped_release>(),
             "T12c: byte size of serialized state for seq_id (llama_state_seq_get_size).")
        .def("state_seq_get_data",
             [](bwr::Context & self, llama_seq_id seq_id) {
                 auto v = self.state_seq_get_data(seq_id);
                 return py::bytes(reinterpret_cast<const char *>(v.data()), v.size());
             },
             py::arg("seq_id"), py::call_guard<py::gil_scoped_release>(),
             "T12c: serialized KV+state for seq_id as bytes (llama_state_seq_get_data).")
        .def("state_seq_set_data",
             [](bwr::Context & self, llama_seq_id seq_id, py::bytes data) {
                 std::string s = data;
                 std::vector<uint8_t> v(s.begin(), s.end());
                 return self.state_seq_set_data(seq_id, v);
             },
             py::arg("seq_id"), py::arg("data"), py::call_guard<py::gil_scoped_release>(),
             "T12c: restore serialized state into seq_id, returns bytes consumed (llama_state_seq_set_data).")
        .def("expert_activations", &bwr::Context::expert_activations,
             "Drain this decode's recorded MoE router frames (consume "
             "semantics: reading clears). Empty unless built with record_experts.")
        .def("probe_metal_write", &bwr::Context::probe_metal_write,
             "Spike gate for SSD fetch: Metal shared buffers are CPU-writable on UMA.")
        .def("fetch_expert", &bwr::Context::fetch_expert,
             py::arg("layer"), py::arg("expert_idx"),
             "Simulate SSD fetch of expert slab (0.35ms, T11c spike).")
        .def_property_readonly("decode_calls", &bwr::Context::decode_calls)
        .def_property_readonly("n_ctx",     &bwr::Context::n_ctx)
        .def_property_readonly("n_batch",   &bwr::Context::n_batch)
        .def_property_readonly("n_ubatch",  &bwr::Context::n_ubatch)
        .def_property_readonly("n_seq_max", &bwr::Context::n_seq_max)
        .def_property_readonly("n_ctx_seq", &bwr::Context::n_ctx_seq)
        .def_property_readonly("n_rs_seq", &bwr::Context::n_rs_seq)
        .def_property_readonly("kv_unified", &bwr::Context::kv_unified)
        .def("close", &bwr::Context::close,
             "Release the llama_context and its samplers now. Idempotent; required for "
             "deterministic server shutdown (see Context::close in context.h).")
        .def_property_readonly("closed", &bwr::Context::closed);
}
