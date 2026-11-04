// The ask–tell protocol: a search as a pure state machine that never calls its objective.
//
// A search algorithm is a state together with two maps (decision 0003 §2.2):
//
//   ask(limit)  → at most `limit` candidates the algorithm can offer now, each with a ticket;
//   tell(told)  → the state after the evaluations of some outstanding tickets.
//
// The algorithm does not know who evaluates, where, or how long it takes: a function on this thread, a pool of
// threads, a simulation on a remote worker, an instrument, a quantum device. That keeps the search pure, as the
// infrastructure's core must be, while its evaluations touch the world at the edges.
//
// Semantics, which every algorithm of heuresis keeps:
// - Tickets are unique within a run: make_ticket(island, serial), the island in the top 16 bits, so that
//   individuals keep their identity when they migrate. A candidate's origin records the operation that made
//   it (a tag, draws.hpp) and its parents' tickets; a run's individuals form a genealogy, a directed acyclic
//   graph whose sources were sampled from nothing or injected from outside.
// - tell accepts any subset of the outstanding tickets, in any order. An unknown or repeated ticket is refused
//   (std::invalid_argument): an evaluation is never applied twice.
// - A generational algorithm offers a whole generation, possibly in parts when `limit` is smaller, and offers
//   nothing more until every candidate of it has been told. It then updates once, reading the evaluations in
//   ticket order, so its states do not depend on the order in which the evaluations came back.
// - An asynchronous (steady-state) algorithm offers candidates whenever asked, and updates at each tell in the
//   order told; its run is reproducible from the order of arrival, which a record keeps. What it breeds from
//   its current state it offers at most `breadth` at a time (a parameter, by default 1, the classical steady
//   state): a driver without a cap would otherwise breed its whole budget from the initial state at once.
// - stopped() says the algorithm has nothing more to offer by its own criteria (converged, exhausted); budgets
//   are the driver's (run.hpp).
// - inject(individuals) hands an algorithm evaluated points from outside: migrants from another island, a
//   restart's archive, proposals made by something the library does not know (a person, a language model, an
//   agent through the API). emigrants(k) gives k of its individuals to send.

#pragma once

#include <heuresis/search/evaluation.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace heuresis::search {

using Ticket = std::uint64_t;

constexpr Ticket make_ticket(std::uint32_t island, std::uint64_t serial) { return (static_cast<Ticket>(island & 0xFFFFu) << 48) | (serial & 0xFFFFFFFFFFFFull); }
constexpr std::uint32_t island_of(Ticket t) { return static_cast<std::uint32_t>(t >> 48); }
constexpr std::uint64_t serial_of(Ticket t) { return t & 0xFFFFFFFFFFFFull; }

inline constexpr std::size_t unlimited = std::numeric_limits<std::size_t>::max();

// What made a candidate: an operation (a tag) and its parents.
struct Origin {
  std::uint32_t operation = 0;
  std::vector<Ticket> parents;
};

template <class G>
struct Candidate {
  Ticket ticket = 0;
  G genotype{};
  Origin origin;
};

// The evaluation of an outstanding ticket.
struct Told {
  Ticket ticket = 0;
  Evaluation evaluation;
};

// An evaluated candidate.
template <class G>
struct Individual {
  Ticket ticket = 0;
  G genotype{};
  Evaluation evaluation;
  Origin origin;
};

template <class A>
concept Optimiser = requires(A& a, const A& c, std::size_t limit, std::span<const Told> told) {
  typename A::Genotype;
  { a.ask(limit) } -> std::same_as<std::vector<Candidate<typename A::Genotype>>>;
  a.tell(told);
  { c.stopped() } -> std::convertible_to<bool>;
};

template <class A>
concept Injectable = Optimiser<A> && requires(A& a, std::span<const Individual<typename A::Genotype>> in) { a.inject(in); };

template <class A>
concept Migratory = Injectable<A> && requires(const A& c, std::size_t k) {
  { c.emigrants(k) } -> std::same_as<std::vector<Individual<typename A::Genotype>>>;
};

// The bookkeeping of one generation: its candidates, handed out in order, and their evaluations as they come
// back. Note holds what the algorithm keeps beside each candidate (a sampled direction, a parent's index).
template <class G, class Note = char>
class Batch {
 public:
  bool active() const { return !candidates_.empty(); }
  bool complete() const { return active() && told_ == candidates_.size(); }
  std::size_t size() const { return candidates_.size(); }
  std::size_t outstanding() const { return handed_ - told_; }

  void open(std::vector<Candidate<G>> candidates, std::vector<Note> notes = {}) {
    if (active()) throw std::logic_error("heuresis::search::Batch: a generation is still open");
    candidates_ = std::move(candidates);
    notes_ = std::move(notes);
    if (notes_.empty()) notes_.resize(candidates_.size());
    evaluations_.assign(candidates_.size(), Evaluation{});
    received_.assign(candidates_.size(), 0);
    index_.clear();
    for (std::size_t i = 0; i < candidates_.size(); i++) index_[candidates_[i].ticket] = i;
    handed_ = told_ = 0;
  }

  // The next candidates not yet handed out, at most `limit`.
  std::vector<Candidate<G>> take(std::size_t limit) {
    std::vector<Candidate<G>> out;
    while (handed_ < candidates_.size() && out.size() < limit) out.push_back(candidates_[handed_++]);
    return out;
  }

  // Records an evaluation; true when it completes the generation.
  bool accept(const Told& t) {
    const auto it = index_.find(t.ticket);
    if (it == index_.end() || it->second >= handed_) throw std::invalid_argument("heuresis::search::Batch: a ticket that is not outstanding");
    if (received_[it->second]) throw std::invalid_argument("heuresis::search::Batch: a ticket told twice");
    received_[it->second] = 1;
    evaluations_[it->second] = t.evaluation;
    told_++;
    return complete();
  }

  // Whether the ticket belongs to this generation and is awaited.
  bool awaits(Ticket t) const {
    const auto it = index_.find(t);
    return it != index_.end() && it->second < handed_ && !received_[it->second];
  }

  const Candidate<G>& candidate(std::size_t i) const { return candidates_[i]; }
  const Evaluation& evaluation(std::size_t i) const { return evaluations_[i]; }
  const Note& note(std::size_t i) const { return notes_[i]; }
  std::span<const Evaluation> evaluations() const { return evaluations_; }

  // The generation's individuals, in ticket order.
  std::vector<Individual<G>> individuals() const {
    std::vector<Individual<G>> out(candidates_.size());
    for (std::size_t i = 0; i < out.size(); i++) out[i] = {candidates_[i].ticket, candidates_[i].genotype, evaluations_[i], candidates_[i].origin};
    return out;
  }

  void close() {
    candidates_.clear();
    notes_.clear();
    evaluations_.clear();
    received_.clear();
    index_.clear();
    handed_ = told_ = 0;
  }

 private:
  std::vector<Candidate<G>> candidates_;
  std::vector<Note> notes_;
  std::vector<Evaluation> evaluations_;
  std::vector<char> received_;
  std::unordered_map<Ticket, std::size_t> index_;
  std::size_t handed_ = 0, told_ = 0;
};

// The candidates outstanding with an asynchronous algorithm: handed out, not yet told.
template <class G, class Note = char>
class Pending {
 public:
  void add(Candidate<G> c, Note note = {}) { items_.emplace(c.ticket, Item{std::move(c), std::move(note)}); }
  std::size_t size() const { return items_.size(); }
  bool contains(Ticket t) const { return items_.count(t) != 0; }
  // Removes the ticket and returns its candidate and note; refuses one that is not outstanding.
  std::pair<Candidate<G>, Note> take(Ticket t) {
    const auto it = items_.find(t);
    if (it == items_.end()) throw std::invalid_argument("heuresis::search::Pending: a ticket that is not outstanding");
    auto out = std::make_pair(std::move(it->second.candidate), std::move(it->second.note));
    items_.erase(it);
    return out;
  }

 private:
  struct Item {
    Candidate<G> candidate;
    Note note;
  };
  std::unordered_map<Ticket, Item> items_;
};

// The indices 0..n−1 ordered best first by `better`, ties by index: the deterministic ranking of a generation.
inline std::vector<std::size_t> ranking(std::span<const Evaluation> e) {
  std::vector<std::size_t> order(e.size());
  for (std::size_t i = 0; i < order.size(); i++) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return better(e[a], e[b]); });
  return order;
}

}  // namespace heuresis::search
