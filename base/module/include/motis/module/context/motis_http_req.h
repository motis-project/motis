#pragma once

#include "net/http/client/client.h"
#include "net/http/client/https_client.h"

#include "motis/module/context/get_io_service.h"

namespace motis::module {

using http_future_t =
    std::shared_ptr<ctx::future<ctx_data, net::http::client::response>>;

http_future_t motis_http_req_impl(char const* src_location,
                                  net::http::client::request req);

#define motis_http(req) motis::module::motis_http_req_impl(CTX_LOCATION, req)

}  // namespace motis::module
