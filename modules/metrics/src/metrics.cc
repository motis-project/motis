#include "motis/metrics/metrics.h"

#include "prometheus/registry.h"
#include "prometheus/text_serializer.h"

namespace mm = motis::module;
namespace fbs = flatbuffers;

namespace motis::metrics {

metrics::metrics() : module("Metrics", "metrics") {}

metrics::~metrics() noexcept = default;

void metrics::init(mm::registry& reg) {
  reg.register_op("/metrics",
                  [&](mm::msg_ptr const& msg) { return request(msg); }, {});
}

mm::msg_ptr metrics::request(mm::msg_ptr const&) const {
  auto registry = get_shared_data<std::shared_ptr<prometheus::Registry>>(
                      to_res_id(mm::global_res_id::METRICS))
                      .get();

  auto const serializer = prometheus::TextSerializer{};
  auto const metrics_str = serializer.Serialize(registry->Collect());

  mm::message_creator mc;

  auto headers = std::vector<fbs::Offset<HTTPHeader>>{};
  headers.emplace_back(
      CreateHTTPHeader(mc, mc.CreateString("Content-Type"),
                       mc.CreateString("text/plain; version=0.0.4")));

  auto payload = mc.CreateString(metrics_str);

  mc.create_and_finish(
      MsgContent_HTTPResponse,
      CreateHTTPResponse(mc, HTTPStatus_OK, mc.CreateVector(headers), payload)
          .Union());

  return make_msg(mc);
}

}  // namespace motis::metrics
