#include "motis/module/event_collector.h"

#include "boost/filesystem.hpp"

#include "fmt/ranges.h"

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/module/clog_redirect.h"
#include "motis/module/context/motis_publish.h"

namespace fs = boost::filesystem;

namespace motis::module {

event_collector::event_collector(std::string data_dir, std::string module_name,
                                 registry& reg, import_op_t op)
    : data_dir_{std::move(data_dir)},
      module_name_{std::move(module_name)},
      reg_{reg},
      op_{std::move(op)},
      progress_tracker_{
          utl::get_global_progress_trackers().get_tracker(module_name_)} {
  progress_tracker_->status("WAITING").show_progress(false);
}

event_collector* event_collector::require(
    std::string const& name, std::function<bool(msg_ptr)> matcher) {
  auto const matcher_it =
      matchers_.emplace(dependency_matcher{name, std::move(matcher)});
  waiting_for_.emplace(name);
  reg_.subscribe(
      "/import",
      [&, matcher_it = matcher_it.first, name,
       self = shared_from_this()](msg_ptr const& msg) -> msg_ptr {
        auto const logs_path = fs::path{data_dir_} / "log";
        fs::create_directories(logs_path);
        clog_redirect redirect{
            (logs_path / (module_name_ + ".txt")).generic_string().c_str()};

        // Dummy message asking for initial status.
        // Send "waiting for" dependencies list.
        if (msg->get()->content_type() == MsgContent_MotisSuccess) {
          progress_tracker_->status(fmt::format("WAITING: {}", waiting_for_));
          return nullptr;
        }

        // Check message type.
        if (!(matcher_it->matcher_fn_)(msg)) {
          return nullptr;
        }

        // Prevent double execution.
        if (executed_) {
          LOG(logging::info) << "prevented double execution";
          LOG(logging::info) << "previous import events\n";
          for (auto const& [k, v] : dependencies_) {
            LOG(logging::info) << k << ": " << v->to_json(true);
          }
          LOG(logging::info) << "new import event: " << msg->to_json(true);
          return nullptr;
        }

        dependencies_[name] = msg;
        waiting_for_.erase(name);
        if (!waiting_for_.empty()) {
          progress_tracker_->status(fmt::format("WAITING: {}", waiting_for_));
          return nullptr;  // Still waiting for a message.
        }

        // All messages arrived -> start.
        activate_progress_tracker(progress_tracker_);
        progress_tracker_->status("RUNNING").show_progress(true);
        try {
          executed_ = true;
          op_(dependencies_);
          progress_tracker_->status("FINISHED").show_progress(false);
        } catch (std::exception const& e) {
          progress_tracker_->status(fmt::format("ERROR: {}", e.what()))
              .show_progress(false);
        } catch (...) {
          progress_tracker_->status("ERROR: UNKNOWN EXCEPTION")
              .show_progress(false);
        }

        return nullptr;
      });
  return this;
}

}  // namespace motis::module
