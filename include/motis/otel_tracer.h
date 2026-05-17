#pragma once

#include "opentelemetry/sdk/resource/resource.h"

#include "motis/config.h"
#include "motis/otel_runtime_context.h"

namespace motis {

void init_opentelemetry(config::otlp const&);

void init_opentelemetry_tracer_http(
    opentelemetry::sdk::resource::Resource const&, config::otlp const&);

}  // namespace motis