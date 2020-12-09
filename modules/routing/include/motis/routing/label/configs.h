#pragma once

#include "motis/routing/label/comparator.h"
#include "motis/routing/label/criteria/absurdity.h"
#include "motis/routing/label/criteria/accessibility.h"
#include "motis/routing/label/criteria/late_connections.h"
#include "motis/routing/label/criteria/load_sum.h"
#include "motis/routing/label/criteria/no_intercity.h"
#include "motis/routing/label/criteria/perceived_travel_time.h"
#include "motis/routing/label/criteria/transfers.h"
#include "motis/routing/label/criteria/travel_time.h"
#include "motis/routing/label/criteria/weighted.h"
#include "motis/routing/label/dominance.h"
#include "motis/routing/label/filter.h"
#include "motis/routing/label/initializer.h"
#include "motis/routing/label/label.h"
#include "motis/routing/label/tie_breakers.h"
#include "motis/routing/label/updater.h"

namespace motis::routing {

template <search_dir Dir>
using default_label =
    label<Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
          label_data<travel_time, transfers, absurdity>,
          initializer<travel_time_initializer, transfers_initializer,
                      absurdity_initializer>,
          updater<travel_time_updater, transfers_updater, absurdity_updater>,
          filter<travel_time_filter, transfers_filter>,
          dominance<absurdity_tb, travel_time_dominance, transfers_dominance>,
          dominance<absurdity_post_search_tb, travel_time_alpha_dominance,
                    transfers_dominance>,
          comparator<transfers_dominance>>;

template <search_dir Dir>
using default_simple_label = label<
    Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
    label_data<travel_time, transfers>,
    initializer<travel_time_initializer, transfers_initializer>,
    updater<travel_time_updater, transfers_updater>,
    filter<travel_time_filter, transfers_filter>,
    dominance<default_tb, travel_time_dominance, transfers_dominance>,
    dominance<post_search_tb, travel_time_alpha_dominance, transfers_dominance>,
    comparator<transfers_dominance>>;

template <search_dir Dir>
using single_criterion_label =
    label<Dir, MAX_WEIGHTED, false, get_weighted_lb, label_data<weighted>,
          initializer<weighted_initializer>, updater<weighted_updater>,
          filter<weighted_filter>, dominance<default_tb, weighted_dominance>,
          dominance<post_search_tb>, comparator<weighted_dominance>>;

template <search_dir Dir>
using single_criterion_no_intercity_label =
    label<Dir, MAX_WEIGHTED, false, get_weighted_lb, label_data<weighted>,
          initializer<weighted_initializer>, updater<weighted_updater>,
          filter<weighted_filter, no_intercity_filter>,
          dominance<default_tb, weighted_dominance>, dominance<post_search_tb>,
          comparator<weighted_dominance>>;

template <search_dir Dir>
using late_connections_label = label<
    Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
    label_data<travel_time, transfers, late_connections, absurdity>,
    initializer<travel_time_initializer, transfers_initializer,
                late_connections_initializer, absurdity_initializer>,
    updater<travel_time_updater, transfers_updater, late_connections_updater,
            absurdity_updater>,
    filter<travel_time_filter, transfers_filter, late_connections_filter>,
    dominance<absurdity_tb, travel_time_dominance, transfers_dominance,
              late_connections_dominance>,
    dominance<absurdity_post_search_tb, travel_time_alpha_dominance,
              transfers_dominance, late_connections_post_search_dominance>,
    comparator<transfers_dominance>>;

template <search_dir Dir>
using late_connections_label_for_tests = label<
    Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
    label_data<travel_time, transfers, late_connections>,
    initializer<travel_time_initializer, transfers_initializer,
                late_connections_initializer>,
    updater<travel_time_updater, transfers_updater, late_connections_updater>,
    filter<travel_time_filter, transfers_filter, late_connections_filter>,
    dominance<default_tb, travel_time_dominance, transfers_dominance,
              late_connections_dominance>,
    dominance<post_search_tb, travel_time_alpha_dominance, transfers_dominance,
              late_connections_post_search_dominance_for_tests>,
    comparator<transfers_dominance>>;

template <search_dir Dir>
using accessibility_label =
    label<Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
          label_data<travel_time, transfers, accessibility, absurdity>,
          initializer<travel_time_initializer, transfers_initializer,
                      accessibility_initializer, absurdity_initializer>,
          updater<travel_time_updater, transfers_updater, accessibility_updater,
                  absurdity_updater>,
          filter<travel_time_filter, transfers_filter>,
          dominance<absurdity_tb, travel_time_dominance, transfers_dominance,
                    accessibility_dominance>,
          dominance<absurdity_post_search_tb, travel_time_alpha_dominance,
                    transfers_dominance, accessibility_dominance>,
          comparator<transfers_dominance, accessibility_dominance>>;

#ifdef MOTIS_CAPACITY_IN_SCHEDULE
template <search_dir Dir>
using perceived_travel_time_label =
    label<Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
          label_data<travel_time, transfers, perceived_travel_time, absurdity>,
          initializer<travel_time_initializer, transfers_initializer,
                      perceived_travel_time_initializer, absurdity_initializer>,
          updater<travel_time_updater, transfers_updater,
                  perceived_travel_time_updater, absurdity_updater>,
          filter<travel_time_filter, transfers_filter>,
          dominance<absurdity_tb, travel_time_dominance, transfers_dominance,
                    perceived_travel_time_dominance>,
          dominance<absurdity_post_search_tb, transfers_dominance,
                    perceived_travel_time_dominance>,
          comparator<transfers_dominance, perceived_travel_time_dominance>>;

template <search_dir Dir>
using load_sum_label =
    label<Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
          label_data<travel_time, transfers, load_sum, absurdity>,
          initializer<travel_time_initializer, transfers_initializer,
                      load_sum_initializer, absurdity_initializer>,
          updater<travel_time_updater, transfers_updater, load_sum_updater,
                  absurdity_updater>,
          filter<travel_time_filter, transfers_filter>,
          dominance<absurdity_tb, travel_time_dominance, transfers_dominance,
                    load_sum_dominance>,
          dominance<absurdity_post_search_tb, transfers_dominance,
                    load_sum_dominance>,
          comparator<transfers_dominance, load_sum_dominance>>;
#endif

}  // namespace motis::routing
