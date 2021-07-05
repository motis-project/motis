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

  struct incoming_edge_bucket {
    friend fws_graph;

    using size_type = size_type;
    using value_type = edge_type;

    template <bool Const>
    struct edge_iterator {
      friend incoming_edge_bucket;
      using iterator_category = std::random_access_iterator_tag;
      using value_type = edge_type;
      using difference_type = int;
      using pointer = value_type*;
      using reference = value_type&;
      using const_reference = value_type const&;

      using bwd_bucket_t = typename edge_fws_multimap<
          edge_type>::bwd_mm_t::template bucket<Const>;
      using bucket_iterator_t = typename bwd_bucket_t::const_iterator;

      template <bool IsConst = Const, typename = std::enable_if_t<IsConst>>
      explicit edge_iterator(edge_iterator<false> const& it)
          : edges_{it.edges_}, bucket_it_{it.bucket_it_} {}

      const_reference operator*() const { return edges_.data_[*bucket_it_]; }

      template <bool IsConst = Const, typename = std::enable_if_t<!IsConst>>
      reference operator*() {
        return const_cast<edge_fws_multimap<edge_type>&>(edges_)  // NOLINT
            .data_[*bucket_it_];
      }

      const_reference operator->() const { return edges_.data_[*bucket_it_]; }

      template <bool IsConst = Const, typename = std::enable_if_t<!IsConst>>
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

    friend iterator begin(incoming_edge_bucket const& b) { return b.begin(); }
    friend iterator end(incoming_edge_bucket const& b) { return b.end(); }

    size_type size() const { return edges_.bwd_[to_node_].size(); }
    bool empty() const { return size() == 0; }

    edge_type& operator[](size_type index) { return edges_[to_node_][index]; }

    edge_type& at(size_type index) const { return edges_[to_node_].at(index); }

  protected:
    incoming_edge_bucket(edge_fws_multimap<edge_type>& edges, size_type to_node)
        : edges_{edges}, to_node_{to_node} {}

    edge_fws_multimap<edge_type>& edges_;
    size_type to_node_;
  };

  edge_type& push_back_edge(edge_type const& e) {
    auto const data_index = edges_[e.from_].push_back(e);
    edges_.bwd_[e.to_].push_back(data_index);
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

  size_type node_index(node_type const* node) const {
    return std::distance(nodes_.begin(), node);
  }

  typename edge_fws_multimap<edge_type>::mutable_bucket outgoing_edges(
      size_type from_node) {
    return edges_[from_node];
  }

  typename edge_fws_multimap<edge_type>::const_bucket outgoing_edges(
      size_type from_node) const {
    return edges_[from_node];
  }

  typename edge_fws_multimap<edge_type>::mutable_bucket outgoing_edges(
      node_type const* from_node) {
    return outgoing_edge(node_index(from_node));
  }

  typename edge_fws_multimap<edge_type>::const_bucket outgoing_edges(
      node_type const* from_node) const {
    return outgoing_edge(node_index(from_node));
  }

  incoming_edge_bucket incoming_edges(size_type to_node) {
    return {edges_, to_node};
  }

  incoming_edge_bucket incoming_edges(node_type const* to_node) {
    return incoming_edges(node_index(to_node));
  }

  mcd::vector<node_type> nodes_;
  edge_fws_multimap<edge_type> edges_;
};

}  // namespace motis
