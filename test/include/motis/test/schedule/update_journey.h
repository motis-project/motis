#pragma once

#include <ctime>
#include <string>

#include "motis/core/schedule/event_type.h"

#include "motis/module/message.h"
#include "motis/loader/loader_options.h"

namespace motis::test::schedule::update_journey {

static loader::loader_options dataset_opt{{"test/schedule/update_journey"},
                                          "20161124"};

motis::module::msg_ptr get_free_text_ris_msg(
    std::string const& eva_num, int service_num, event_type type,
    time_t schedule_time, std::string const& trip_start_eva,
    time_t trip_start_schedule_time, int free_text_code,
    std::string const& free_text_text, std::string const& free_text_type);

motis::module::msg_ptr get_routing_request(time_t from, time_t to,
                                           std::string const& eva_from,
                                           std::string const& eva_to);

motis::module::msg_ptr get_delay_ris_msg(std::string const& eva_num,
                                         int service_num, event_type type,
                                         time_t schedule_time,
                                         time_t update_time,
                                         std::string const& trip_start_eva,
                                         time_t trip_start_schedule_time);

motis::module::msg_ptr get_canceled_train_ris_message(
    std::string const& eva_num, int service_num, event_type type,
    time_t schedule_time, std::string const& eva_num1, event_type type1,
    time_t schedule_time1, std::string const& trip_start_eva,
    time_t trip_start_schedule_time);

motis::module::msg_ptr get_track_ris_msg(std::string const& eva_num,
                                         int service_num, event_type type,
                                         time_t schedule_time,
                                         std::string const& trip_start_eva,
                                         time_t trip_start_schedule_time,
                                         std::string const& updated_track);

}  // namespace motis::test::schedule::update_journey
