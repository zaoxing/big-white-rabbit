// pybind11 entry point for the FreeToken-Mac Metal engine.
//
// Every call that can block for a meaningful time (model load, decode) releases the GIL.
// That is load-bearing for the single-process control-plane design: uvicorn's event loop
// and the decode loop share this process, so a decode that held the GIL would stall
// /health and every concurrent request.

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "context.h"
#include "model.h"

namespace py = pybind11;

PYBIND11_MODULE(_freetoken_metal, m) {
    m.doc() = "FreeToken-Mac: llama.cpp/Metal bindings (Phase 0)";

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
        .def_readwrite("flash_attn",      &ftm::ContextParams::flash_attn);

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
        .def_property_readonly("path",        &ftm::Model::path)
        .def_property_readonly("n_ctx_train", &ftm::Model::n_ctx_train)
        .def_property_readonly("n_embd",      &ftm::Model::n_embd)
        .def_property_readonly("n_layer",     &ftm::Model::n_layer)
        .def_property_readonly("n_vocab",     &ftm::Model::n_vocab)
        .def_property_readonly("size_bytes",  &ftm::Model::size_bytes)
        .def_property_readonly("n_params",    &ftm::Model::n_params)
        .def_property_readonly("desc",        &ftm::Model::desc)
        .def("__repr__", [](const ftm::Model & self) {
            return "<freetoken_mac.Model '" + self.desc() + "'>";
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
        .def("memory_seq_rm", &ftm::Context::memory_seq_rm,
             py::arg("seq_id"), py::arg("p0") = -1, py::arg("p1") = -1)
        .def_property_readonly("n_ctx",    &ftm::Context::n_ctx)
        .def_property_readonly("n_batch",  &ftm::Context::n_batch)
        .def_property_readonly("n_ubatch", &ftm::Context::n_ubatch);
}
