// pybind11 entry point for the FreeToken-Mac Metal engine.
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

PYBIND11_MODULE(_freetoken_metal, m) {
    // pybind11's default translators already map std::invalid_argument and
    // std::length_error (Batch's overflow guard) onto Python ValueError, so every
    // bounds check added in Phase 1 reaches Python as an exception, not an abort().
    m.doc() = "FreeToken-Mac: llama.cpp/Metal bindings (Phase 1)";

    m.def("backend_init", &ftm::backend_init_once,
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

    py::class_<ftm::ModelParams>(m, "ModelParams")
        .def(py::init<>())
        .def_readwrite("n_gpu_layers", &ftm::ModelParams::n_gpu_layers)
        .def_readwrite("load_mode",    &ftm::ModelParams::load_mode)
        .def_readwrite("lazy_mode",    &ftm::ModelParams::lazy_mode)
        .def_readwrite("no_alloc",     &ftm::ModelParams::no_alloc);

    py::class_<ftm::ContextParams>(m, "ContextParams")
        .def(py::init<>())
        .def_readwrite("n_ctx",           &ftm::ContextParams::n_ctx)
        .def_readwrite("n_batch",         &ftm::ContextParams::n_batch)
        .def_readwrite("n_ubatch",        &ftm::ContextParams::n_ubatch)
        .def_readwrite("n_seq_max",       &ftm::ContextParams::n_seq_max)
        .def_readwrite("n_threads",       &ftm::ContextParams::n_threads)
        .def_readwrite("n_threads_batch", &ftm::ContextParams::n_threads_batch)
        .def_readwrite("flash_attn",      &ftm::ContextParams::flash_attn)
        // Unified KV buffer: required for a partial-range memory_seq_cp (prefix fork).
        .def_readwrite("kv_unified",      &ftm::ContextParams::kv_unified)
        // Recurrent-state rollback snapshots for partial memory_seq_rm on hybrids.
        .def_readwrite("n_rs_seq",        &ftm::ContextParams::n_rs_seq);

    py::class_<ftm::SamplerParams>(m, "SamplerParams")
        .def(py::init<>())
        .def_readwrite("temp",  &ftm::SamplerParams::temp)
        .def_readwrite("top_k", &ftm::SamplerParams::top_k)
        .def_readwrite("top_p", &ftm::SamplerParams::top_p)
        .def_readwrite("seed",  &ftm::SamplerParams::seed);

    py::class_<ftm::Model, std::shared_ptr<ftm::Model>>(m, "Model")
        .def(py::init<const std::string &, const ftm::ModelParams &>(),
             py::arg("path"), py::arg("params") = ftm::ModelParams(),
             py::call_guard<py::gil_scoped_release>())
        .def("tokenize", &ftm::Model::tokenize,
             py::arg("text"), py::arg("add_special") = true, py::arg("parse_special") = true)
        .def("token_to_piece", &ftm::Model::token_to_piece,
             py::arg("token"), py::arg("special") = false)
        .def("detokenize", &ftm::Model::detokenize,
             py::arg("tokens"), py::arg("unparse_special") = false)
        .def("is_eog", &ftm::Model::is_eog, py::arg("token"))
        .def("meta_val", &ftm::Model::meta_val, py::arg("key"))
        .def("apply_chat_template", &ftm::Model::apply_chat_template,
             py::arg("messages"), py::arg("add_assistant") = true,
             "Render [(role, content), ...] into a prompt via the model's own template.")
        .def_property_readonly("chat_template", &ftm::Model::chat_template)
        .def_property_readonly("path",        &ftm::Model::path)
        .def_property_readonly("n_ctx_train", &ftm::Model::n_ctx_train)
        .def_property_readonly("n_embd",      &ftm::Model::n_embd)
        .def_property_readonly("n_layer",     &ftm::Model::n_layer)
        .def_property_readonly("n_vocab",     &ftm::Model::n_vocab)
        .def_property_readonly("size_bytes",  &ftm::Model::size_bytes)
        .def_property_readonly("n_params",    &ftm::Model::n_params)
        .def_property_readonly("desc",        &ftm::Model::desc)
        .def("close", &ftm::Model::close,
             "Free the weights now. Close every Context on this model FIRST; required "
             "for deterministic shutdown (see Model::close in model.h).")
        .def_property_readonly("closed", &ftm::Model::closed)
        .def("__repr__", [](const ftm::Model & self) {
            return self.closed() ? std::string("<freetoken_mac.Model closed>")
                                 : "<freetoken_mac.Model '" + self.desc() + "'>";
        });

    py::class_<ftm::Batch>(m, "Batch")
        .def(py::init<int32_t, int32_t>(),
             py::arg("capacity"), py::arg("n_seq_max_per_token") = 1)
        .def("clear", &ftm::Batch::clear)
        .def("add", &ftm::Batch::add,
             py::arg("token"), py::arg("pos"), py::arg("seq_id"), py::arg("logits") = false,
             "Append a token; returns its batch row index (raises past capacity).")
        .def("add_shared", &ftm::Batch::add_shared,
             py::arg("token"), py::arg("pos"), py::arg("seq_ids"), py::arg("logits") = false)
        .def_property_readonly("n_tokens", &ftm::Batch::n_tokens)
        .def_property_readonly("capacity", &ftm::Batch::capacity)
        .def_property_readonly("n_seq_max_per_token", &ftm::Batch::n_seq_max_per_token)
        .def_property_readonly("max_seq_id", &ftm::Batch::max_seq_id)
        .def_property_readonly("max_pos", &ftm::Batch::max_pos)
        .def("__len__", &ftm::Batch::n_tokens)
        .def("__repr__", [](const ftm::Batch & self) {
            return "<freetoken_mac.Batch " + std::to_string(self.n_tokens()) + "/" +
                   std::to_string(self.capacity()) + " tokens>";
        });

    py::class_<ftm::Context>(m, "Context")
        .def(py::init<std::shared_ptr<ftm::Model>, const ftm::ContextParams &, const ftm::SamplerParams &>(),
             py::arg("model"),
             py::arg("params")  = ftm::ContextParams(),
             py::arg("sampler") = ftm::SamplerParams(),
             py::call_guard<py::gil_scoped_release>())
        .def("decode_seq0", &ftm::Context::decode_seq0, py::arg("tokens"),
             py::call_guard<py::gil_scoped_release>())
        .def("sample_last", &ftm::Context::sample_last,
             py::call_guard<py::gil_scoped_release>())
        .def("accept", &ftm::Context::accept, py::arg("token"))
        // Batched step. The caller's reference keeps `batch` alive for the whole call,
        // so releasing the GIL cannot let Python free the arrays llama_decode reads.
        .def("decode", &ftm::Context::decode, py::arg("batch"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_seq_sampler", &ftm::Context::set_seq_sampler,
             py::arg("seq_id"), py::arg("sampler") = ftm::SamplerParams())
        .def("reset_seq_sampler", &ftm::Context::reset_seq_sampler, py::arg("seq_id"))
        .def("has_seq_sampler", &ftm::Context::has_seq_sampler, py::arg("seq_id"))
        .def("sample_seq", &ftm::Context::sample_seq, py::arg("seq_id"), py::arg("idx"),
             py::call_guard<py::gil_scoped_release>())
        .def("accept_seq", &ftm::Context::accept_seq, py::arg("seq_id"), py::arg("token"))
        .def("row_has_logits", &ftm::Context::row_has_logits, py::arg("idx"))
        // sample_last()'s precondition, exposed so a caller can ask instead of catching.
        .def_property_readonly("any_row_has_logits", &ftm::Context::any_row_has_logits)
        .def("memory_seq_rm", &ftm::Context::memory_seq_rm,
             py::arg("seq_id"), py::arg("p0") = -1, py::arg("p1") = -1,
             "Drop a KV range; returns llama.cpp's verdict. A partial rm can "
             "report False (hybrid without rollback snapshots) -- callers that "
             "packed past the rewind point must check.")
        .def("memory_seq_cp", &ftm::Context::memory_seq_cp,
             py::arg("src"), py::arg("dst"), py::arg("p0") = -1, py::arg("p1") = -1)
        .def("memory_seq_keep", &ftm::Context::memory_seq_keep, py::arg("seq_id"))
        .def_property_readonly("decode_calls", &ftm::Context::decode_calls)
        .def_property_readonly("n_ctx",     &ftm::Context::n_ctx)
        .def_property_readonly("n_batch",   &ftm::Context::n_batch)
        .def_property_readonly("n_ubatch",  &ftm::Context::n_ubatch)
        .def_property_readonly("n_seq_max", &ftm::Context::n_seq_max)
        .def_property_readonly("n_ctx_seq", &ftm::Context::n_ctx_seq)
        .def_property_readonly("n_rs_seq", &ftm::Context::n_rs_seq)
        .def_property_readonly("kv_unified", &ftm::Context::kv_unified)
        .def("close", &ftm::Context::close,
             "Release the llama_context and its samplers now. Idempotent; required for "
             "deterministic server shutdown (see Context::close in context.h).")
        .def_property_readonly("closed", &ftm::Context::closed);
}
