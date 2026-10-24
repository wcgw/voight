//! The candle crates the detector runs on: those from crates.io, or with the
//! `wgpu` feature the fork that adds a wgpu backend, which candle hasn't
//! released yet (huggingface/candle#3379).

#[cfg(not(feature = "wgpu"))]
pub use {candle_core as core, candle_nn as nn, candle_transformers as transformers};
#[cfg(feature = "wgpu")]
pub use {
    candle_core_wgpu as core, candle_nn_wgpu as nn, candle_transformers_wgpu as transformers,
};
