#pragma once

#include <memory>
#include <vector>

#include "guess/guesser.h"

#include "motis/module/module.h"

namespace motis::guesser {

struct guesser : public motis::module::module {
  guesser();
  void init(motis::module::registry&) override;

private:
  motis::module::msg_ptr guess(motis::module::msg_ptr const&);

  std::vector<unsigned> station_indices_;
  std::unique_ptr<guess::guesser> guesser_;
};

}  // namespace motis::guesser
