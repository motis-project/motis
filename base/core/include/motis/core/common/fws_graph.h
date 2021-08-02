#pragma once

#include <iterator>

#include "motis/core/common/dynamic_fws_multimap.h"
#include "motis/vector.h"

namespace motis {

template <typename T, typename SizeType = std::uint32_t>
struct edge_fws_multimap
    : public dynamic_fws_multimap_base<edge_fws_multimap<T, SizeType>, T,
                                       SizeType> {
  using bwd_mm_t = dynamic_fws_multimap<SizeType>;

  void entries_moved(SizeType const map_index, SizeType const old_data_index,
                     SizeType const new_data_index, SizeType const count) {
    auto const old_data_end = old_data_index + count;
    for (auto const& e : this->at(map_index)) {
      for (auto& i : bwd_.at(e.to_)) {
        if (i >= old_data_index && i < old_data_end) {
          i = new_data_index + (i - old_data_index);
        }
      }
    }
  }

  bwd_mm_t bwd_;
};

template <typename Node, typename Edge>
struct fws_graph {
  using node_type = Node;
  using edge_type = Edge;
  using size_type = std::uint32_t;

  // fwd: node_id -> outgoing edges
  // bwd: node_id -> trace (edge indices) -> incoming edges

  template <bool Const>
  struct incoming_edge_bucket {
    friend fws_graph;

    using value_type = edge_type;

    template <bool IsConst = Const, typename = std::enable_if_t<IsConst>>
    explicit incoming_edge_bucket(incoming_edge_bucket<false> const& b)
        : edges_{b.edges_}, to_node_{b.to_node_} {}

    template <bool ConstIt>
    struct edge_iterator {
      friend incoming_edge_bucket;
      using iterator_category = std::random_access_iterator_tag;
      using value_type = edge_type;
      using difference_type = int;
      using pointer = value_type*;
      using reference = value_type&;
      using const_reference = value_type const&;

      using bwd_bucket_t = typename edge_fws_multimap<
          edge_type>::bwd_mm_t::template bucket<ConstIt>;
      using bucket_iterator_t = typename bwd_bucket_t::const_iterator;

      template <bool IsConst = ConstIt, typename = std::enable_if_t<IsConst>>
      explicit edge_iterator(edge_iterator<false> const& it)
          : edges_{it.edges_}, bucket_it_{it.bucket_it_} {}

      const_reference operator*() const { return edges_.data_[*bucket_it_]; }

      template <bool IsConst = ConstIt, typename = std::enable_if_t<!IsConst>>
      reference operator*() {
        return const_cast<edge_fws_multimap<edge_type>&>(edges_)  // NOLINT
            .data_[*bucket_it_];
      }

      const_reference operator->() const { return edges_.data_[*bucket_it_]; }

      template <bool IsConst = ConstIt, typename = std::enable_if_t<!IsConst>>
      reference operator->() {
        return const_cast<edge_fws_multimap<edge_type>&>(edges_)  // NOLINT
            .data_[*bucket_it_];
      }

      edge_iterator& operator+=(difference_type n) {
        bucket_it_ += n;
        return *this;
      }

      edge_iterator& operator-=(difference_type n) {
        bucket_it_ -= n;
        return *this;
      }

      edge_iterator& operator++() {
        ++bucket_it_;
        return *this;
      }

      edge_iterator& operator--() {
        ++bucket_it_;
        return *this;
      }

      edge_iterator operator+(difference_type n) const {
        return {edges_, bucket_it_ + n};
      }

      edge_iterator operator-(difference_type n) const {
        return {edges_, bucket_it_ - n};
      }

      int operator-(edge_iterator const& rhs) const {
        return bucket_it_ - rhs.bucket_it_;
      };

      bool operator<(edge_iterator const& rhs) const {
        return bucket_it_ < rhs.bucket_it_;
      }
      bool operator<=(edge_iterator const& rhs) const {
        return bucket_it_ <= rhs.bucket_it_;
      }
      bool operator>(edge_iterator const& rhs) const {
        return bucket_it_ > rhs.bucket_it_;
      }
      bool operator>=(edge_iterator const& rhs) const {
        return bucket_it_ >= rhs.bucket_it_;
      }

      bool operator==(edge_iterator const& rhs) const {
        return &edges_ == &rhs.edges_ && bucket_it_ == rhs.bucket_it_;
      }

      bool operator!=(edge_iterator const& rhs) const {
        return &edges_ != &rhs.edges_ || bucket_it_ != rhs.bucket_it_;
      }

    protected:
      edge_iterator(edge_fws_multimap<edge_type> const& edges,
                    bucket_iterator_t bucket_it)
          : edges_(edges), bucket_it_(bucket_it) {}

      edge_fws_multimap<edge_type> const& edges_;
      bucket_iterator_t bucket_it_;
    };

    using iterator = edge_iterator<false>;
    using const_iterator = edge_iterator<true>;

    iterator begin() { return {edges_, edges_.bwd_[to_node_].begin()}; }

    const_iterator begin() const {
      return {edges_, edges_.bwd_[to_node_].cbegin()};
    }

    iterator end() { return {edges_, edges_.bwd_[to_node_].end()}; }

    const_iterator end() const {
      return {edges_, edges_.bwd_[to_node_].cend()};
    }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    friend iterator begin(incoming_edge_bucket& b) { return b.begin(); }
    friend const_iterator begin(incoming_edge_bucket const& b) {
      return b.begin();
    }

    friend iterator end(incoming_edge_bucket& b) { return b.end(); }
    friend const_iterator end(incoming_edge_bucket const& b) { return b.end(); }

    size_type size() const { return edges_.bwd_[to_node_].size(); }
    [[nodiscard]] bool empty() const { return size() == 0; }

    edge_type const& operator[](size_type index) const {
      return edges_.data_[edges_.bwd_[to_node_][index]];
    }

    template <bool IsConst = Const, typename = std::enable_if_t<!IsConst>>
    edge_type& operator[](size_type index) {
      return mutable_edges().data_[edges_.bwd_[to_node_][index]];
    }

    edge_type const& at(size_type index) const {
      return edges_.data_.at(edges_.bwd_[to_node_].at(index));
    }

    template <bool IsConst = Const, typename = std::enable_if_t<!IsConst>>
    edge_type& at(size_type index) {
      return mutable_edges().data_.at(edges_.bwd_[to_node_].at(index));
    }

    edge_type const& front() const { return (*this)[0]; }
    edge_type& front() { return (*this)[0]; }

    edge_type const& back() const {
      assert(!empty());
      return (*this)[size() - 1];
    }

    edge_type& back() {
      assert(!empty());
      return (*this)[size() - 1];
    }

  protected:
    incoming_edge_bucket(edge_fws_multimap<edge_type> const& edges,
                         size_type to_node)
        : edges_{edges}, to_node_{to_node} {}

    edge_fws_multimap<edge_type>& mutable_edges() {
      return const_cast<edge_fws_multimap<edge_type>&>(edges_);  // NOLINT
    }

    edge_fws_multimap<edge_type> const& edges_;
    size_type to_node_;
  };

  using mutable_outgoing_edge_bucket =
      typename edge_fws_multimap<edge_type>::mutable_bucket;
  using const_outgoing_edge_bucket =
      typename edge_fws_multimap<edge_type>::const_bucket;

  using mutable_incoming_edge_bucket = incoming_edge_bucket<false>;
  using const_incoming_edge_bucket = incoming_edge_bucket<true>;

  edge_type& push_back_edge(edge_type const& e) {
    auto const data_index = edges_[e.from_].push_back(e);
    edges_.bwd_[e.to_].push_back(data_index);
    return edges_.data_[data_index];
  }

  edge_type& push_back_edge(edge_type&& e) {
    auto const from = e.from_;
    auto const to = e.to_;
    auto const data_index = edges_[from].emplace_back(std::move(e));
    edges_.bwd_[to].push_back(data_index);
    return edges_.data_[data_index];
  }

  template <typename... Args>
  edge_type& emplace_back_edge(size_type from_node, size_type to_node,
                               Args&&... args) {
    auto const data_index =
        edges_[from_node].emplace_back(std::forward(args)...);
    edges_.bwd_[to_node].push_back(data_index);
    return edges_.data_[data_index];
  }

  template <typename... Args>
  node_type& emplace_back_node(Args&&... args) {
    auto& n = nodes_.emplace_back(std::forward<Args>(args)...);
    init_node_edges(n);
    return n;
  }

  node_type& push_back_node(node_type const& node) {
    auto& n = nodes_.push_back(node);
    init_node_edges(n);
    return n;
  }

  size_type node_index(node_type const* node) const {
    return std::distance(nodes_.begin(), node);
  }

  void init_node_edges(node_type const& node) {
    auto const index = node_index(&node);
    edges_[index];
    edges_.bwd_[index];
  }

  mutable_outgoing_edge_bucket outgoing_edges(size_type from_node) {
    return edges_[from_node];
  }

  const_outgoing_edge_bucket outgoing_edges(size_type from_node) const {
    return edges_[from_node];
  }

  mutable_outgoing_edge_bucket outgoing_edges(node_type const* from_node) {
    return outgoing_edge(node_index(from_node));
  }

  const_outgoing_edge_bucket outgoing_edges(node_type const* from_node) const {
    return outgoing_edge(node_index(from_node));
  }

  mutable_incoming_edge_bucket incoming_edges(size_type to_node) {
    return {edges_, to_node};
  }

  const_incoming_edge_bucket incoming_edges(size_type to_node) const {
    return {edges_, to_node};
  }

  mutable_incoming_edge_bucket incoming_edges(node_type const* to_node) {
    return incoming_edges(node_index(to_node));
  }

  const_incoming_edge_bucket incoming_edges(node_type const* to_node) const {
    return incoming_edges(node_index(to_node));
  }

  std::size_t node_count() const { return nodes_.size(); }
  std::size_t edge_count() const { return edges_.element_count(); }

  mcd::vector<node_type> nodes_;
  edge_fws_multimap<edge_type> edges_;
};

}  // namespace motis
