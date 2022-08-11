#pragma once

#include "motis/mcraptor/raptor_query.h"
#include "motis/mcraptor/raptor_timetable.h"
#include "bag.h"
#include "label.h"

namespace motis::mcraptor {

struct rounds {
public:
  bag* bags_{nullptr};
  stop_id stop_count_{invalid<stop_id>};

  explicit rounds(stop_id const stop_count)
      : stop_count_{stop_count} {
    bags_ = new bag[this->byte_size()];
    this->reset();
  }

  rounds() = delete;
  rounds(rounds const&) = delete;
  rounds(rounds const&&) = delete;
  rounds& operator=(rounds const&) = delete;
  rounds& operator=(rounds const&&) = delete;
  ~rounds() {
    delete[] bags_;
  }

  size_t byte_size() const {
    size_t const number_of_entries =
        static_cast<size_t>(max_raptor_round) * stop_count_;
    size_t const size_in_bytes = sizeof(bag) * number_of_entries;
    return size_in_bytes;
  }

  void reset() const {
    size_t const number_of_entries = byte_size() / sizeof(bag);
    //    std::fill(bags, bags + number_of_entries, Bag());
    for(int i = 0; i < number_of_entries; ++i) {
      bags_[i] = bag();
    }
  }

  bag* data() const { return bags_; }

  bag* operator[](raptor_round const index) {  // NOLINT
    return &bags_[static_cast<ptrdiff_t>(index) * stop_count_];
  };

  bag const* operator[](raptor_round const index) const {
    return &bags_[static_cast<ptrdiff_t>(index) * stop_count_];
  };

  std::vector<label> getAllLabelsForStop(stop_id stop_id, raptor_round max_round) {
    std::vector<label> res = std::vector<label>();
    for(int i = 0; i < max_round; i++) {
      bag& bag = (*this)[i][stop_id];
      if(bag.is_valid()) {
        for(label& l : bag.labels_) {
          res.push_back(l);
        }
      }
    }
    return res;
  }
};

} // namespace motis::mcraptor