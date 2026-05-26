#include "motis/otel_tracer.h"

#include <memory>
#include <utility>

#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_factory.h"
#include "opentelemetry/sdk/trace/exporter.h"
#include "opentelemetry/sdk/trace/processor.h"
#include "opentelemetry/sdk/trace/samplers/always_on.h"
#include "opentelemetry/sdk/trace/samplers/always_on_factory.h"
#include "opentelemetry/sdk/trace/simple_processor.h"
#include "opentelemetry/sdk/trace/simple_processor_factory.h"
#include "opentelemetry/sdk/trace/tracer_provider.h"
#include "opentelemetry/sdk/trace/tracer_provider_factory.h"
#include "opentelemetry/trace/propagation/http_trace_context.h"
#include "opentelemetry/trace/provider.h"

namespace motis {

void init_opentelemetry_tracer(
    opentelemetry::sdk::resource::Resource const& resource,
    config::otlp const& c) {
  namespace sdktrace = opentelemetry::sdk::trace;

  // TODO
  // create otlp opts from config - or pass otlp opts directly

  auto opts = opentelemetry::exporter::otlp::OtlpHttpExporterOptions{};
  opts.url = c.url_;

  auto exporter =
      opentelemetry::exporter::otlp::OtlpHttpExporterFactory::Create(opts);

  auto processor =
      opentelemetry::sdk::trace::SimpleSpanProcessorFactory::Create(
          std::move(exporter));

  auto sampler = opentelemetry::sdk::trace::AlwaysOnSamplerFactory::Create();

  // When e.g. net gets named Tracer does this is created as child to this
  // provider?
  auto provider =
      std::shared_ptr{opentelemetry::sdk::trace::TracerProviderFactory::Create(
          std::move(processor), resource, std::move(sampler))};
  opentelemetry::trace::Provider::SetTracerProvider(provider);
}

void init_opentelemetry(config::otlp const& c,
                        std::string_view const motis_version) {
  auto resource_attributes = opentelemetry::sdk::resource::ResourceAttributes{
      {"service.name", "motis"}, {"service.version", motis_version}};
  auto resource =
      opentelemetry::sdk::resource::Resource::Create(resource_attributes);

  opentelemetry::context::RuntimeContext::SetRuntimeContextStorage(
      std::make_shared<otel_runtime_context_storage>());

  if (c.http_) {
    init_opentelemetry_tracer(resource, c);
  }

  opentelemetry::context::propagation::GlobalTextMapPropagator::
      SetGlobalPropagator(
          std::make_shared<
              opentelemetry::trace::propagation::HttpTraceContext>());
}

}  // namespace motis