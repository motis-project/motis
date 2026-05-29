#pragma once

#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/tracer.h"

#include "motis/config.h"

namespace motis {

void init_opentelemetry(config::otlp const&, std::string_view const);

void cleanup_opentelemetry_tracer();

inline opentelemetry::nostd::shared_ptr<opentelemetry::trace::Tracer>
get_otel_tracer() {
  return opentelemetry::trace::Provider::GetTracerProvider()->GetTracer(
      "motis");
}

}  // namespace motis