#include "motis/module/event_collector.h"

#include "boost/filesystem.hpp"

#include "utl/verify.h"

#include "motis/module/clog_redirect.h"
#include "motis/module/context/motis_publish.h"

namespace fs = boost::filesystem;

namespace motis::module {

event_collector::event_collector(std::string data_dir, std::string name,
                                 registry& reg, import_op_t op)
    : data_dir_{std::move(data_dir)},
      module_name_{std::move(name)},
      reg_{reg},
      op_{std::move(op)} {}

void event_collector::update_status(motis::import::Status const status,
                                    uint8_t const progress) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_StatusUpdate,
      motis::import::CreateStatusUpdate(
          fbb, fbb.CreateString(module_name_),
          fbb.CreateVector(utl::to_vec(
              waiting_for_, [&](auto&& w) { return fbb.CreateString(w); })),
          status, fbb.CreateString(""), progress)
          .Union(),
      "/import", DestinationType_Topic);
  ctx::await_all(motis_publish(make_msg(fbb)));
}

void event_collector::require(std::string const& name,
                              std::function<bool(msg_ptr)> matcher) {
  auto const matcher_it =
      matchers_.emplace(dependency_matcher{name, std::move(matcher)});
  waiting_for_.emplace(name);
  reg_.subscribe(
      "/import",
      [&, matcher_it = matcher_it.first, name,
       self = shared_from_this()](msg_ptr const& msg) -> msg_ptr {
        auto const logs_path = fs::path{data_dir_} / "log";
        fs::create_directories(logs_path);
        log_streambuf redirect{
            module_name_,
            (logs_path / (module_name_ + ".txt")).generic_string().c_str()};

        // Dummy message asking for initial status.
        // Send "waiting for" dependencies list.
        if (msg->get()->content_type() == MsgContent_MotisSuccess) {
          update_status(motis::import::Status_WAITING);
          return nullptr;
        }

        if (!(matcher_it->matcher_fn_)(msg)) {
          return nullptr;  // Not the right message type.
        }

        dependencies_[name] = msg;
        waiting_for_.erase(name);
        if (!waiting_for_.empty()) {
          return nullptr;  // Still waiting for a message.
        }

        // All messages arrived -> start.
        update_status(motis::import::Status_RUNNING);
        try {
          op_(dependencies_);
          update_status(motis::import::Status_FINISHED);
        } catch (std::exception const& e) {
          std::clog << '\0' << 'E' << e.what() << '\0';
        }

        return nullptr;
      });
}

}  // namespace motis::module
