#pragma once

#include <memory>
#include <vector>

#include "guess/guesser.h"

#include "motis/module/module.h"

namespace motis::guesser {

struct guesser : public motis::module::module {
  guesser();
  void init(motis::module::registry&) override;
  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override;

private:
  void update_stations();
  motis::module::msg_ptr guess(motis::module::msg_ptr const&);

  std::vector<unsigned> station_indices_;
  std::unique_ptr<guess::guesser> guesser_;

  bool import_successful_{false};
};

}  // namespace motis::guesser
