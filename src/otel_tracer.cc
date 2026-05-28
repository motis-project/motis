#include "motis/otel_tracer.h"

#include <chrono>
#include <memory>
#include <utility>

#include "opentelemetry/nostd/shared_ptr.h"
#include "utl/verify.h"

#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/exporters/otlp/otlp_http.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_factory.h"
#include "opentelemetry/sdk/resource/resource.h"
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

#include "motis/otel_runtime_context.h"

namespace motis {

void init_opentelemetry_tracer(
    opentelemetry::sdk::resource::Resource const& resource,
    config::otlp const& c) {

  auto const& http_opts = c.http_.value();
  auto opts = opentelemetry::exporter::otlp::OtlpHttpExporterOptions{};
  opts.url = http_opts.url_;
  if (http_opts.content_type_ == "json") {
    opts.content_type =
        opentelemetry::exporter::otlp::HttpRequestContentType::kJson;
  } else if (http_opts.content_type_ == "binary") {
    opts.content_type =
        opentelemetry::exporter::otlp::HttpRequestContentType::kBinary;
  } else {
    utl::fail("Invalid OTLP content type {}", http_opts.content_type_);
  }
  opts.use_json_name = http_opts.use_json_name_;
  opts.timeout = std::chrono::seconds(c.timeout_);
  for (auto [key, value] : c.headers_) {
    opts.http_headers.insert({key, value});
  }

  auto exporter =
      opentelemetry::exporter::otlp::OtlpHttpExporterFactory::Create(opts);

  auto processor =
      opentelemetry::sdk::trace::SimpleSpanProcessorFactory::Create(
          std::move(exporter));

  auto sampler = opentelemetry::sdk::trace::AlwaysOnSamplerFactory::Create();

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

void cleanup_opentelemetry_tracer() {
  if (auto const provider = opentelemetry::trace::Provider::GetTracerProvider();
      provider) {
    provider->ForceFlush();
  }

  auto const none = std::shared_ptr<opentelemetry::trace::NoopTracerProvider>();
  opentelemetry::trace::Provider::SetTracerProvider(none);
}

}  // namespace motis