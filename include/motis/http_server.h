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
                                 std::string path,
                                 H&& handler) {
  app.get(std::move(path),
          [&app, h = std::move(handler)](uWS::HttpResponse<SSL>* res,
                                         uWS::HttpRequest* req) {
            send_response(res, h(boost::urls::url_view{req->getFullUrl()}));
          });
}

template <GETJsonHandler H, bool SSL>
void handle_get(uWS::CachingApp<SSL>& app, std::string path, H&& handler) {
  handle_get_generic_response(
      app, std::move(path),
      [h = std::move(handler)](boost::urls::url_view const& url) {
        auto response = http_response{};
        response.body() =
            boost::json::serialize(boost::json::value_from(h(url)));
        return response;
      });
}

template <typename App>
struct local_cluster {
  template <typename Fn>
  explicit local_cluster(
      Fn&& init, std::size_t n_threads = std::thread::hardware_concurrency())
      : apps_{n_threads}, threads_{n_threads} {
    for (auto i = 0U; i != n_threads; ++i) {
      threads_.emplace_back([this, i, &init]() {
        {
          auto const lock = std::scoped_lock{init_mutex_};
          apps_[i] = init();
        }
        apps_[i].run();
      });
    }
  }

  ~local_cluster() {
    for (auto& t : threads_) {
      t.join();
    }
  }

  template <typename... Args>
  local_cluster& listen(Args&&... args) {
    main_app_.listen(std::forward<Args>(args)...);
    return *this;
  }

  local_cluster& run(uWS::SocketContextOptions options) {
    main_app_ = App{options};
    main_app_->preOpen(
        [&](struct us_socket_context_t*,
            LIBUS_SOCKET_DESCRIPTOR fd) -> LIBUS_SOCKET_DESCRIPTOR {
          auto& worker = apps_[(++round_robin_ % apps_.size())];
          worker.getLoop()->defer(
              [fd, worker_ptr = &worker]() { worker_ptr->adoptSocket(fd); });
          return static_cast<LIBUS_SOCKET_DESCRIPTOR>(-1);
        });
    main_app_.run();
  }

  App main_app_;
  std::vector<App> apps_;
  std::vector<std::thread> threads_;
  std::mutex init_mutex_;
  std::atomic_uint round_robin_;
};

}  // namespace motis