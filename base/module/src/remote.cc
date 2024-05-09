#include "motis/module/remote.h"

#include <map>
#include <vector>

#include "boost/asio/deadline_timer.hpp"
#include "boost/asio/ssl/context.hpp"
#include "boost/date_time/posix_time/posix_time_duration.hpp"

#include "utl/to_vec.h"

#include "net/wss_client.h"

#include "motis/core/common/logging.h"

namespace motis::module {

using req_id_t = int32_t;

struct remote::impl : std::enable_shared_from_this<impl> {
  impl(registry& reg, boost::asio::io_service& ios,  //
       std::string host, std::string port,  //
       std::function<void()> on_register, std::function<void()> on_unregister)
      : reg_{reg},
        ios_{ios},
        host_{std::move(host)},
        port_{std::move(port)},
        on_register_{std::move(on_register)},
        on_unregister_{std::move(on_unregister)} {
    boost::system::error_code ignore;
    (void)ctx_.set_verify_mode(boost::asio::ssl::verify_none, ignore);
  }

  void stop() {
    stopped_ = true;
    timer_.cancel();
    if (ws_) {
      ws_->stop();
    }
    for (auto const& timeout : timeouts_) {
      timeout->cancel();
    }
  }

  void start() {
    if (stopped_) {
      return;
    }

    ws_ = std::make_unique<net::wss_client>(ios_, ctx_, host_, port_);
    ws_->on_fail(
        [self = shared_from_this()](boost::system::error_code const& ec) {
          self->on_fail(ec);
        });
    ws_->on_msg(
        [self = shared_from_this()](std::string const& raw, bool binary) {
          self->on_msg(raw, binary);
        });
    ws_->run([self = shared_from_this()](auto&& cb) { self->on_connect(cb); });
  }

  void on_fail(boost::system::error_code ec) {
    if (stopped_) {
      return;
    }

    LOG(logging::error) << "connection to " << host_ << ":" << port_
                        << " failed (" << ec.message() << "), retry in 5s";
    for (auto const& m : methods_) {
      LOG(logging::info) << "remote " << host_ << ":" << port_
                         << " unregistered for " << m;
    }
    schedule_restart();
    reg_.unregister_remote_op(methods_);
    methods_.clear();

    if (on_unregister_) {
      on_unregister_();
    }
  }

  void on_msg(std::string const& raw, bool binary) {
    if (stopped_) {
      return;
    }

    auto msg = binary ? make_msg(raw.data(), raw.size()) : make_msg(raw);
    switch (msg->get()->content_type()) {
      case MsgContent_ApiDescription:
        methods_ = reg_.register_remote_ops(
            utl::to_vec(*motis_content(ApiDescription, msg)->methods(),
                        [](auto&& s) { return s->str(); }),
            ios_.wrap([this](msg_ptr const& msg, callback cb) {
              send(msg, std::move(cb));
            }));

        if (on_register_) {
          on_register_();
        }

        for (auto const& m : methods_) {
          LOG(logging::info)
              << "remote " << host_ << ":" << port_ << " registered for " << m;
        }

        if (on_register_) {
          on_register_();
        }
        break;

      default:
        if (auto const it = pending_.find(msg->id()); it != end(pending_)) {
          it->second(std::move(msg), {});
        } else {
          LOG(logging::error) << "unknown incoming message of type "
                              << EnumNameMsgContent(msg->get()->content_type());
        }
        break;
    }
  }

  void on_connect(boost::system::error_code ec) {
    if (ec) {
      LOG(logging::error) << "failed to connect to " << host_ << ":" << port_
                          << " (" << ec.message() << "), retry in 5s";
      schedule_restart();
    } else {
      ws_->send(make_no_msg("/api", ++next_req_id_)->to_string(), true);
    }
  }

  void send(msg_ptr const& msg, callback cb) {
    if (stopped_) {
      return;
    }

    ++next_req_id_;

    auto timer = *timeouts_
                      .emplace(std::make_shared<boost::asio::deadline_timer>(
                          ios_, boost::posix_time::seconds{60}))
                      .first;

    timer->async_wait([timer, id = next_req_id_, self = shared_from_this()](
                          boost::system::error_code ec) {
      if (auto const it = self->pending_.find(id); it != end(self->pending_)) {
        if (ec != boost::asio::error::operation_aborted) {
          LOG(logging::error) << "timeout for operation " << id;
          return it->second(
              nullptr, std::make_error_code(std::errc::operation_canceled));
        }
        self->pending_.erase(it);
      }
      self->timeouts_.erase(timer);
    });

    pending_.emplace(next_req_id_, [c = std::move(cb), timer](
                                       msg_ptr res, std::error_code ec) {
      c(std::move(res), ec);
      timer->cancel();
    });

    msg->get()->mutate_id(next_req_id_);
    ws_->send(msg->to_string(), true);
  }

  void schedule_restart() {
    timer_.expires_from_now(boost::posix_time::seconds{5});
    timer_.async_wait([me = shared_from_this()](boost::system::error_code ec) {
      if (!me->stopped_ && ec != boost::asio::error::operation_aborted) {
        me->start();
      }
    });
  }

  registry& reg_;
  boost::asio::io_service& ios_;
  boost::asio::deadline_timer timer_{ios_};
  std::set<std::shared_ptr<boost::asio::deadline_timer>> timeouts_;
  boost::asio::ssl::context ctx_{boost::asio::ssl::context::sslv23};
  std::string host_, port_;
  std::unique_ptr<net::wss_client> ws_;
  bool stopped_{false};
  std::map<req_id_t, callback> pending_;
  std::vector<std::string> methods_;
  req_id_t next_req_id_{0};
  std::function<void()> on_register_, on_unregister_;
};

remote::remote(registry& reg, boost::asio::io_service& ios,  //
               std::string const& host, std::string const& port,  //
               std::function<void()> const& on_register,
               std::function<void()> const& on_unregister)
    : impl_{std::make_shared<impl>(reg, ios, host, port, on_register,
                                   on_unregister)} {}

void remote::send(msg_ptr const& msg, callback cb) const {
  impl_->send(msg, std::move(cb));
}

void remote::stop() const { impl_->stop(); }

void remote::start() const { impl_->start(); }

}  // namespace motis::module
