#include "motis/otel_tracer.h"

#include <memory>
#include <utility>

#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/exporters/otlp_http_exporter.h"
#include "opentelemetry/sdk/trace/exporter.h"
#include "opentelemetry/sdk/trace/processor.h"
#include "opentelemetry/sdk/trace/samplers/always_on.h"
#include "opentelemetry/sdk/trace/simple_processor.h"
#include "opentelemetry/trace/propagation/http_trace_context.h"
#include "opentelemetry/trace/tracer_provider.h"

namespace motis {

void init_opentelemetry(config::otlp const& opts) {
  auto resource_attributes = opentelemetry::sdk::resource::ResourceAttributes{
      {"service.name", "motis"},
      {
          "service.version" /* TODO add version*/
      }};
  auto resource =
      opentelemetry::sdk::resource::Resource::Create(resource_attributes);

  opentelemetry::context::RuntimeContext::SetRuntimeContextStorage(
      std::make_shared<otel_runtime_context_storage>());

  if (opts.otlp_http_) {
    init_opentelemetry_tracer(resource, opts);
  }

  // What is this used for exactly?
  opentelemetry::context::propagation::GlobalTextMapPropagator::
      SetGlobalPropagator(
          std::make_shared<
              opentelemetry::trace::propagation::HttpTraceContext>());
}

void init_opentelemetry_tracer(
    opentelemetry::sdk::resource::Resource const& resource,
    config::otlp const& c) {

  // TODO
  // create otlp opts from config - or pass otlp opts directly

  auto const opts = opentelemetry::exporter::otlp::OtlpHttpExporterOptions {
    .url = c.otlp_url
  }

  auto exporter = std::unique_ptr<opentelemetry::sdk::trace::SpanExporter>(
      new opentelemetry::exporter::otlp::OtlpHttpExporter(opts));

  auto procesor = std::unique_ptr<opentelemtry::sdk::trace::SpanProcessor>(
      new opentelemetry::sdk::trace::SimpleSpanProcessor(std::move(exporter)));

  auto sampler = std::unique_ptr<opentelemetry::sdk::trace::AlwaysOnSampler>(
      new opentelemetry::sdk::trace::AlwaysOnSampler);

  // When e.g. net gets named Tracer does this is created as child to this
  // provider?
  auto provider = nostd::shared_ptr<opentelemetry::sdk::trace::TracerProvider>(
      std::move(procesor), resource, std::move(sampler));
  opentelemtry::trace::provider::SetTracerProvider(provider);
}

}  // namespace motis