#include "motis/core/journey/generate_journey_transport.h"

#include "motis/core/access/service_access.h"

namespace motis {

journey::transport generate_journey_transport(
    unsigned int from, unsigned int to, connection_info const* con_info,
    schedule const& sched, duration duration, int mumo_id, unsigned mumo_price,
    unsigned mumo_accessibility) {
  bool is_walk = true;
  mcd::string name;
  mcd::string cat_name;
  unsigned cat_id = 0;
  unsigned clasz = 0;
  unsigned train_nr = 0;
  mcd::string line_identifier;
  mcd::string direction;
  mcd::string provider;

  if (con_info != nullptr) {
    is_walk = false;

    cat_id = con_info->family_;
    cat_name = sched.categories_[con_info->family_]->name_;

    auto clasz_it = sched.classes_.find(cat_name);
    clasz = static_cast<service_class_t>(clasz_it == end(sched.classes_)
                                             ? service_class::OTHER
                                             : clasz_it->second);

    line_identifier = con_info->line_identifier_;

    train_nr =
        output_train_nr(con_info->train_nr_, con_info->original_train_nr_);

    if (con_info->dir_ != nullptr) {
      direction = *con_info->dir_;
    }

    if (con_info->provider_ != nullptr) {
      provider = con_info->provider_->full_name_;
    }

    name = get_service_name(sched, con_info);
  }

  return {from,
          to,
          is_walk,
          name.str(),
          cat_name.str(),
          cat_id,
          clasz,
          train_nr,
          line_identifier.str(),
          duration,
          mumo_id,
          direction.str(),
          provider.str(),
          mumo_price,
          mumo_accessibility,
          ""};
}

}  // namespace motis
