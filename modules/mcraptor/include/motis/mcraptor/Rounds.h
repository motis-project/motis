#pragma once

#include "Bag.h"
#include "Label.h"
#include "motis/mcraptor/raptor_timetable.h"

namespace motis::mcraptor {

class Rounds {
protected:
  Bag* bags{nullptr};

public:
  stop_id stop_count_{invalid<stop_id>};

  explicit Rounds(stop_id const stop_count)
      : stop_count_{stop_count} {
    bags = new Bag[this->byte_size()];
    this->reset();
  }

  Rounds() = delete;
  Rounds(Rounds const&) = delete;
  Rounds(Rounds const&&) = delete;
  Rounds& operator=(Rounds const&) = delete;
  Rounds& operator=(Rounds const&&) = delete;
  ~Rounds() {
    delete[] bags;
  }

  size_t byte_size() const {
    size_t const number_of_entries =
        static_cast<size_t>(max_raptor_round) * stop_count_;
    size_t const size_in_bytes = sizeof(Bag) * number_of_entries;
    return size_in_bytes;
  }

  void reset() const {
    size_t const number_of_entries = byte_size() / sizeof(Bag);
//    std::fill(bags, bags + number_of_entries, Bag());
    for(int i = 0; i < number_of_entries; i++) {
      bags[i] = Bag();
    }
  }

  Bag* data() const { return bags; }

  Bag* operator[](raptor_round const index) {  // NOLINT
    return &bags[static_cast<ptrdiff_t>(index) * stop_count_];
  };

  Bag const* operator[](raptor_round const index) const {
    return &bags[static_cast<ptrdiff_t>(index) * stop_count_];
  };
};

} // namespace motis::mcraptor