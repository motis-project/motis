#pragma once

#include <memory>

#include "opentelemetry/trace/tracer.h"

namespace motis {

extern std::shared_ptr<opentelemetry::trace::Tracer> motis_tracer;

}  // namespace motis
