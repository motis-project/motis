#include "motis/nigiri/nigiri.h"

#include "conf/date_time.h"

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/hrd/load_timetable.h"

#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"
#include "motis/nigiri/routing.h"

namespace fs = std::filesystem;
namespace mm = motis::module;
namespace n = ::nigiri;

namespace motis::nigiri {

struct nigiri::impl {
  std::shared_ptr<cista::wrapped<n::timetable>> tt_;
  std::vector<std::string> tags_;
};

nigiri::nigiri() : module("Next Generation Routing", "nigiri") {
  param(no_cache_, "no_cache", "disable timetable caching");
  param(first_day_, "first_day",
        "YYYY-MM-DD, leave empty to use first day in source data");
  param(num_days_, "num_days", "number of days, ignored if first_day is empty");
}

nigiri::~nigiri() = default;

void nigiri::init(motis::module::registry& reg) {
  reg.register_op("/nigiri",
                  [&](mm::msg_ptr const& msg) {
                    return route(impl_->tags_, **impl_->tt_, msg);
                  },
                  {});
}

void nigiri::import(motis::module::import_dispatcher& reg) {
  std::make_shared<mm::event_collector>(
      get_data_directory().generic_string(), "nigiri", reg,
      [this](mm::event_collector::dependencies_map_t const& dependencies,
             mm::event_collector::publish_fn_t const& publish) {
        auto const& msg = dependencies.at("SCHEDULE");

        impl_ = std::make_unique<impl>();

        auto const begin = std::chrono::sys_days{
            std::chrono::duration_cast<std::chrono::days>(std::chrono::seconds{
                first_day_.empty() ? 0U : conf::parse_date_time(first_day_)})};
        auto const interval =
            first_day_.empty()
                ? n::kMaxInterval<std::chrono::sys_days>
                : n::interval<std::chrono::sys_days>{
                      begin, begin + std::chrono::days{num_days_}};

        using import::FileEvent;
        auto h = cista::BASE_HASH;
        h = cista::hash_combine(h, interval.from_.time_since_epoch().count());
        h = cista::hash_combine(h, interval.to_.time_since_epoch().count());

        auto datasets = std::vector<std::tuple<
            n::source_idx_t, decltype(n::loader::hrd::configs)::const_iterator,
            std::unique_ptr<n::loader::dir>>>{};
        for (auto const [i, p] :
             utl::enumerate(*motis_content(FileEvent, msg)->paths())) {
          if (p->tag()->str() != "schedule") {
            continue;
          }
          auto const path = fs::path{p->path()->str()};
          auto d = n::loader::make_dir(path);
          auto const c = utl::find_if(n::loader::hrd::configs, [&](auto&& c) {
            return n::loader::hrd::applicable(c, *d);
          });
          utl::verify(c != end(n::loader::hrd::configs),
                      "no loader applicable to {}", path);
          h = n::loader::hrd::hash(*c, *d, h);

          datasets.emplace_back(n::source_idx_t{i}, c, std::move(d));

          auto const tag = p->options()->str();
          impl_->tags_.emplace_back(tag + (tag.empty() ? "" : "-"));
        }

        auto const data_dir = get_data_directory() / "nigiri";
        auto const dump_file_path = data_dir / fmt::to_string(h);
        if (!no_cache_ && std::filesystem::is_regular_file(dump_file_path)) {
          impl_->tt_ = std::make_shared<cista::wrapped<n::timetable>>(
              n::timetable::read(cista::memory_holder{
                  cista::file{dump_file_path.string().c_str(), "r"}
                      .content()}));
        } else {
          impl_->tt_ = std::make_shared<cista::wrapped<n::timetable>>(
              cista::raw::make_unique<n::timetable>());
          for (auto const& [src, config, dir] : datasets) {
            LOG(logging::info) << "loading nigiri timetable with configuration "
                               << config->version_.view();
            n::loader::hrd::load_timetable(src, *config, *dir, **impl_->tt_,
                                           interval);
          }

          if (!no_cache_) {
            std::filesystem::create_directories(data_dir);
            (*impl_->tt_)->write(dump_file_path);
          }
        }

        add_shared_data(to_res_id(mm::global_res_id::NIGIRI_TIMETABLE),
                        impl_->tt_->get());
        add_shared_data(to_res_id(mm::global_res_id::NIGIRI_TAGS),
                        &impl_->tags_);

        LOG(logging::info) << "nigiri timetable: stations="
                           << (*impl_->tt_)->locations_.names_.size()
                           << ", trips=" << (*impl_->tt_)->trip_debug_.size()
                           << "\n";

        import_successful_ = true;

        mm::message_creator fbb;
        fbb.create_and_finish(MsgContent_NigiriEvent,
                              motis::import::CreateNigiriEvent(fbb).Union(),
                              "/import", DestinationType_Topic);
        publish(make_msg(fbb));
      })
      ->require("SCHEDULE", [](mm::msg_ptr const& msg) {
        if (msg->get()->content_type() != MsgContent_FileEvent) {
          return false;
        }
        using import::FileEvent;
        return utl::all_of(
            *motis_content(FileEvent, msg)->paths(),
            [](import::ImportPath const* p) {
              if (p->tag()->str() != "schedule") {
                return true;
              }
              auto const d = n::loader::make_dir(fs::path{p->path()->str()});
              return utl::any_of(n::loader::hrd::configs, [&](auto&& c) {
                return n::loader::hrd::applicable(c, *d);
              });
            });
      });
}

}  // namespace motis::nigiri
