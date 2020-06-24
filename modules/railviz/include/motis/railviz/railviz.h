#pragma once

#include "motis/module/module.h"

namespace motis::railviz {

struct train_retriever;

struct railviz : public motis::module::module {
  railviz();
  ~railviz() override;

  railviz(railviz const&) = delete;
  railviz& operator=(railviz const&) = delete;

  railviz(railviz&&) = delete;
  railviz& operator=(railviz&&) = delete;

  void init(motis::module::registry&) override;

private:
  motis::module::msg_ptr get_map_config(motis::module::msg_ptr const&);

  static motis::module::msg_ptr get_trip_guesses(motis::module::msg_ptr const&);
  static motis::module::msg_ptr get_station(motis::module::msg_ptr const&);

  motis::module::msg_ptr get_trains(motis::module::msg_ptr const&) const;
  motis::module::msg_ptr get_trips(motis::module::msg_ptr const&);

  std::unique_ptr<train_retriever> train_retriever_;

  std::string initial_permalink_;
  std::string tiles_redirect_;
};

}  // namespace motis::railviz
