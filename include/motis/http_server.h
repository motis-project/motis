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
  app.get(std::move(path), [h = std::move(handler)](uWS::HttpResponse<SSL>* res,
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
      Fn&& init, std::size_t n_threads = std::thread::hardware_concurrency()) {
    for (auto i = 0U; i != n_threads; ++i) {
      threads_.emplace_back([this, i, init]() {
        apps_.emplace_back(init())
            .setUserData(this)
            .preOpen([](struct us_socket_context_t*, LIBUS_SOCKET_DESCRIPTOR fd,
                        void* user_data) -> LIBUS_SOCKET_DESCRIPTOR {
              auto const self = reinterpret_cast<local_cluster*>(user_data);
              auto const worker_idx =
                  (++self->round_robin_ % self->apps_.size());
              std::cout << "Moving socket to worker " << worker_idx
                        << std::endl;
              auto& worker = self->apps_[worker_idx];
              worker.getLoop()->defer([fd, worker_idx, worker_ptr = &worker]() {
                std::cout << "Worker " << worker_idx << " picking up work"
                          << std::endl;
                worker_ptr->adoptSocket(fd);
              });
              return static_cast<LIBUS_SOCKET_DESCRIPTOR>(-1);
            })
            .run();
        std::cout << "App " << i << " terminated" << std::endl;
      });
    }
  }

  ~local_cluster() {
    for (auto& t : threads_) {
      t.join();
    }
  }

  std::vector<App> apps_;
  std::vector<std::thread> threads_;
  std::atomic_uint round_robin_;
};

}  // namespace motis