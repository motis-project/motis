#pragma once

#include "boost/url/url_view.hpp"

#include "boost/json/serialize.hpp"
#include "boost/json/value_from.hpp"
#include "boost/json/value_to.hpp"

#include "App.h"

#include "motis/http_response.h"

namespace motis {

template <typename T>
concept JsonSerializable = boost::json::has_value_to<T>::value;

template <typename T>
concept JsonDeSerializable = boost::json::has_value_to<T>::value;

template <typename Fn>
concept GETJsonHandler = requires(boost::urls::url_view const& url, Fn f) {
  { f(url) } -> JsonSerializable;
};

template <typename Fn>
concept PostJsonHandler = requires(Fn f, typename utl::first_argument<Fn> arg) {
  { f(arg) } -> JsonSerializable;
};

template <typename Fn>
concept HttpHandler = requires(boost::urls::url_view const& url, Fn f) {
  { f(url) } -> std::convertible_to<http_response>;
};

void enable_cors(http_response& res) {
  using namespace boost::beast::http;
  res.insert(field::access_control_allow_origin, "*");
  res.insert(field::access_control_allow_headers, "*");
  res.insert(field::access_control_allow_methods, "GET, POST, OPTIONS");
}

template <bool SSL>
void send_response(uWS::HttpResponse<SSL>* res,
                   http_response&& response,
                   std::shared_ptr<bool>&& is_aborted = {}) {
  enable_cors(response);
  res->cork(
      [res, is_aborted = std::move(is_aborted), r = std::move(response)]() {
        res->writeStatus(uWS::HTTP_200_OK);
        for (auto const& h : r.base()) {
          res->writeHeader(to_string(h.name()), h.value());
        }
        if (is_aborted == nullptr || !*is_aborted) {
          res->end(r.body());
        }
      });
}

template <HttpHandler H, bool SSL>
void handle_get_generic_response(uWS::CachingApp<SSL>& app,
                                 auto& executor,
                                 std::string path,
                                 H&& handler) {
  app.get(
      std::move(path), [&executor, &app, h = std::move(handler)](
                           uWS::HttpResponse<SSL>* res, uWS::HttpRequest* req) {
        auto is_aborted = std::make_shared<bool>(false);
        executor.post([is_aborted, res, req, &h, &app]() mutable {
          auto response = h(boost::urls::url_view{req->getFullUrl()});
          app.getLoop()->defer([res, response = std::move(response),
                                is_aborted = std::move(is_aborted)]() mutable {
            send_response(res, std::move(response), std::move(is_aborted));
          });
        });
        res->onAborted([is_aborted]() { *is_aborted = true; });
      });
}

template <GETJsonHandler H, bool SSL>
void handle_get(uWS::CachingApp<SSL>& app,
                auto& executor,
                std::string path,
                H&& handler) {
  handle_get_generic_response(
      app, executor, std::move(path),
      [h = std::move(handler)](boost::urls::url_view const& url) {
        auto response = http_response{};
        response.body() =
            boost::json::serialize(boost::json::value_from(h(url)));
        return response;
      });
}

}  // namespace motis