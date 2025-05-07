#include "motis/endpoints/metrics.h"

#include <iostream>

#include "prometheus/registry.h"
#include "prometheus/text_serializer.h"

namespace motis::ep {

net::reply metrics::operator()(net::route_request const& req, bool) const {
  auto res = net::web_server::string_res_t{boost::beast::http::status::ok,
                                           req.version()};
  res.insert(boost::beast::http::field::content_type,
             "text/plain; version=0.0.4");
  set_response_body(res, req,
                    prometheus::TextSerializer{}.Serialize(metrics_.Collect()));
  res.keep_alive(req.keep_alive());
  return res;
}

}  // namespace motis::ep