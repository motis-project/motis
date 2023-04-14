#include "motis/nigiri/nigiri.h"

#include "cista/memory_holder.h"

#include "conf/date_time.h"

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/loader.h"
#include "nigiri/loader/hrd/loader.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/print_transport.h"
#include "nigiri/timetable.h"

#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"
#include "motis/nigiri/routing.h"

namespace fs = std::filesystem;
namespace mm = motis::module;
namespace n = ::nigiri;

namespace motis::nigiri {

struct nigiri::impl {
  impl() {
    loaders_.emplace_back(std::make_unique<n::loader::gtfs::gtfs_loader>());
    loaders_.emplace_back(
        std::make_unique<n::loader::hrd::hrd_5_00_8_loader>());
    loaders_.emplace_back(
        std::make_unique<n::loader::hrd::hrd_5_20_26_loader>());
    loaders_.emplace_back(
        std::make_unique<n::loader::hrd::hrd_5_20_39_loader>());
    loaders_.emplace_back(
        std::make_unique<n::loader::hrd::hrd_5_20_avv_loader>());
  }
  std::vector<std::unique_ptr<n::loader::loader_interface>> loaders_;
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
  impl_ = std::make_unique<impl>();
  std::make_shared<mm::event_collector>(
      get_data_directory().generic_string(), "nigiri", reg,
      [this](mm::event_collector::dependencies_map_t const& dependencies,
             mm::event_collector::publish_fn_t const& publish) {
        auto const& msg = dependencies.at("SCHEDULE");

        auto const begin =
            date::sys_days{std::chrono::duration_cast<std::chrono::days>(
                std::chrono::seconds{conf::parse_date_time(first_day_)})};
        auto const interval = n::interval<date::sys_days>{
            begin, begin + std::chrono::days{num_days_}};

        using import::FileEvent;
        auto h = cista::BASE_HASH;
        h = cista::hash_combine(h, interval.from_.time_since_epoch().count());
        h = cista::hash_combine(h, interval.to_.time_since_epoch().count());

        auto datasets =
            std::vector<std::tuple<n::source_idx_t,
                                   decltype(impl_->loaders_)::const_iterator,
                                   std::unique_ptr<n::loader::dir>>>{};
        for (auto const [i, p] :
             utl::enumerate(*motis_content(FileEvent, msg)->paths())) {
          if (p->tag()->str() != "schedule") {
            continue;
          }
          auto const path = fs::path{p->path()->str()};
          auto d = n::loader::make_dir(path);
          auto const c = utl::find_if(
              impl_->loaders_, [&](auto&& c) { return c->applicable(*d); });
          utl::verify(c != end(impl_->loaders_), "no loader applicable to {}",
                      path);
          h = cista::hash_combine((*c)->hash(*d), h);

          datasets.emplace_back(n::source_idx_t{i}, c, std::move(d));

          auto const tag = p->options()->str();
          impl_->tags_.emplace_back(tag + (tag.empty() ? "default_" : "_"));
        }

        auto const data_dir = get_data_directory() / "nigiri";
        auto const dump_file_path = data_dir / fmt::to_string(h);
        if (!no_cache_ && fs::is_regular_file(dump_file_path)) {
          try {
            impl_->tt_ = std::make_shared<cista::wrapped<n::timetable>>(
                n::timetable::read(cista::memory_holder{
                    cista::file{dump_file_path.string().c_str(), "r"}
                        .content()}));
          } catch (std::exception const& e) {
            LOG(logging::error)
                << "cannot read cached timetable image: " << e.what()
                << ", retry loading from scratch";
            goto read_datasets;
          }
        } else {
        read_datasets:
          impl_->tt_ = std::make_shared<cista::wrapped<n::timetable>>(
              cista::raw::make_unique<n::timetable>());

          (*impl_->tt_)->date_range_ = interval;
          n::loader::register_special_stations(**impl_->tt_);

          for (auto const& [src, loader, dir] : datasets) {
            LOG(logging::info) << "loading nigiri timetable with configuration "
                               << (*loader)->name();
            (*loader)->load(src, *dir, **impl_->tt_);
          }

          n::loader::finalize(**impl_->tt_);

          if (!no_cache_) {
            std::filesystem::create_directories(data_dir);
            (*impl_->tt_)->write(dump_file_path);
          }
          //
          //          auto& tt = **impl_->tt_;
          //          auto const reverse = [](std::string s) {
          //            std::reverse(s.begin(), s.end());
          //            return s;
          //          };
          //          auto const range = tt.internal_interval_days();
          //          auto const num_days = static_cast<size_t>(
          //              (range.size() + std::chrono::days{2}) /
          //              std::chrono::days{1});
          //          for (auto i = 0U; i != tt.transport_traffic_days_.size();
          //          ++i) {
          //            auto const transport_idx = n::transport_idx_t{i};
          //            auto const traffic_days =
          //                tt.bitfields_.at(tt.transport_traffic_days_.at(transport_idx));
          //            std::cout << "TRAFFIC_DAYS="
          //                      << reverse(traffic_days.to_string().substr(
          //                             traffic_days.size() - num_days))
          //                      << "\n";
          //            for (auto d = range.from_; d != range.to_;
          //                 d += std::chrono::days{1}) {
          //              auto const day_idx =
          //                  n::day_idx_t{static_cast<n::day_idx_t::value_t>(
          //                      (d - range.from_) / std::chrono::days{1})};
          //              if (traffic_days.test(to_idx(day_idx))) {
          //                date::to_stream(std::cout, "%F", d);
          //                std::cout << " (day_idx=" << day_idx << ")\n";
          //                n::print_transport(tt, std::cout, {transport_idx,
          //                day_idx});
          //              }
          //            }
          //          }
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
      ->require("SCHEDULE", [this](mm::msg_ptr const& msg) {
        if (msg->get()->content_type() != MsgContent_FileEvent) {
          return false;
        }
        using import::FileEvent;
        return utl::all_of(
            *motis_content(FileEvent, msg)->paths(),
            [this](import::ImportPath const* p) {
              if (p->tag()->str() != "schedule") {
                return true;
              }
              auto const d = n::loader::make_dir(fs::path{p->path()->str()});
              return utl::any_of(impl_->loaders_,
                                 [&](auto&& c) { return c->applicable(*d); });
            });
      });
}

}  // namespace motis::nigiri
