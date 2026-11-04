// Random draws keyed by what they are for.
//
// Every random draw of a search is taken from a calx Stream whose identity says what it is for: a purpose
// (sampling, mutating, selecting, migrating; a tag computed from its name), an epoch (a generation, a step)
// and an index (an individual, a parent, a coordinate block). The key of the streams is the run's seed folded
// with the island's number, so that islands of one run draw independently. A draw therefore depends only on
// (seed, island, purpose, epoch, index, position), never on which thread, process or machine makes it, nor
// on the order in which evaluations come back: a run gives the same result on one thread, on many, or spread
// over workers (decision 0003 §2.5).
//
// Tags are 32-bit FNV-1a hashes of names (Fowler, Noll and Vo; the parameters of the IETF draft
// draft-eastlake-fnv), computed at compile time, so a new algorithm declares its own purposes without a
// central list. Two names whose hashes collide would share streams; development/tools/check-tags.mjs hashes
// every name the library uses and finds none that do.

#pragma once

#include <calx/random/stream.hpp>

#include <cstdint>
#include <string_view>

namespace heuresis::search {

// The 32-bit FNV-1a hash of a name.
constexpr std::uint32_t tag(std::string_view name) {
  std::uint32_t h = 2166136261u;
  for (char c : name) {
    h ^= static_cast<std::uint8_t>(c);
    h *= 16777619u;
  }
  return h;
}

// SplitMix64's finaliser (Steele, Lea and Flood 2014, "Fast splittable pseudorandom number generators",
// OOPSLA): a bijection of 64-bit words that spreads every input bit over the output.
constexpr std::uint64_t mix(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

class Draws {
 public:
  explicit Draws(std::uint64_t seed = 0, std::uint32_t island = 0) : seed_(seed), island_(island), key_(mix(seed ^ mix(island))) {}

  // The stream for (purpose, epoch, index), at its beginning.
  calx::random::Stream operator()(std::uint32_t purpose, std::uint64_t epoch, std::uint64_t index) const {
    return calx::random::Stream(key_, purpose, static_cast<std::uint32_t>(epoch ^ (epoch >> 32)), static_cast<std::uint32_t>(index ^ (index >> 32)));
  }

  std::uint64_t seed() const { return seed_; }
  std::uint32_t island() const { return island_; }

 private:
  std::uint64_t seed_;
  std::uint32_t island_;
  std::uint64_t key_;
};

}  // namespace heuresis::search
