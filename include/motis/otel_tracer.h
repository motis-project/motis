#pragma once

#include "opentelemetry/sdk/resource/resource.h"

#include "motis/config.h"
#include "motis/otel_runtime_context.h"

namespace motis {

void init_opentelemetry(config::otlp const&, std::string_view const);

}  // namespace motis