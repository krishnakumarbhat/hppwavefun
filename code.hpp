#pragma once

#include "util/types.hpp"
#include "util/std.hpp"
#include "util/ndarray.hpp"
#include "util/collections.hpp"
#include "util/rand.hpp"
#include "util/hash.hpp"
#include "util/assert.hpp"
#include "util/bitset.hpp"
#include "util/result.hpp"
#include "util/omp.hpp"

namespace wfc {
enum Flags {
    // add rotations of default patterns to pattern set
    FLAGS_ROTATE = 1 << 0,

    // add reflections of default patterns to pattern set
    FLAGS_REFLECT = 1 << 1
};

// which pattern selection function to use
enum class PatternFunction {
    WEIGHTED
};

// which cell selection function to use
enum class NextCellFunction {
    MIN_ENTROPY
};

// behavior of borders for pattern calculation
// EXCLUDE: any pattern which would include a border is not used
// ZERO: borders have the value T(0)
// CLAMP: borders are clamped to their nearest defined value
// WRAP: borders are wrapped around the input data
enum BorderBehavior {
    EXCLUDE,
    ZERO,
    CLAMP,
    WRAP
};

// permute n-dimensional rotations of an array
template <usize N>
struct Rotator {};

template <>
struct Rotator<2> {
    using V = ivec2;

    // permutes rotations of src
    template <typename T, usize S>
    static inline std::array<std::array<T, S * S>, 4> permute(
        const std::array<T, S * S> &src) {
        std::array<std::array<T, S * S>, 4> dst;
        dst[0] = src;
        dst[1] = rotate_ccw<T, S>(dst[0]);
        dst[2] = rotate_ccw<T, S>(dst[1]);
        dst[3] = rotate_ccw<T, S>(dst[2]);
        return dst;
    }

private:
    template <typename T, usize S>
    static inline std::array<T, S * S> rotate(
        const std::array<T, S * S> &src) {
        std::array<T, S * S> dst;
        for (usize i = 0; i < S; i++) {
            for (usize j = 0; j < S; j++) {
                ndarray::at(V(S), &dst[0], { i, j }) =
                    ndarray::at(V(S), &src[0], { S - j - 1, i });
            }
        }
        return dst;
    }
};

// permutes n-dimensional reflections of an array
template <usize N>
struct Reflector {
    using V = math::vec<N, int, math::defaultp>;

    template <typename T, usize S, usize VOL = math::cexp::pow(S, N)>
    static inline auto permute(
        const std::array<T, S * S> &src) {
        std::array<std::array<T, VOL>, math::cexp::pow(2, N)> dst;

        // iterate over possible permutations of axes
        usize i = 0;
        ndarray::each(
            V(2),
            [&](const V &v) {
                const auto which = math::vec<N, bool, math::defaultp>(v);
                dst[i] = src;
                for (usize j = 0; j < N; j++) {
                    if (which[j]) {
                        dst[i] = reflect_axis<T, S>(dst[i], j);
                    }
                }
                i++;
            });

        return dst;
    }

private:
    template <typename T, usize S, usize VOL = math::cexp::pow(S, N)>
    static inline auto reflect_axis(
        const std::array<T, VOL> &src, usize axis) {
        std::array<T, VOL> dst;
        ndarray::each(
            V(S),
            [&](const V &v) {
                V u = v;
                u[axis] = S - u[axis] - 1;
                ndarray::at(V(S), &dst[0], v) =
                    ndarray::at(V(S), &src[0], u);
            });
        return dst;
    }
};

// neighbors in N dimensions
template <usize N>
struct Neighbors {};

template <>
struct Neighbors<2> {
    static constexpr std::array<ivec2, 4> neighbors =
        {
            ivec2(-1, 0),
            ivec2(1, 0),
            ivec2(0, -1),
            ivec2(0, 1),
        };
};

template <>
struct Neighbors<3> {
    static constexpr std::array<ivec3, 9> neighbors =
        {
            ivec3(-1, 0, 0),
            ivec3(1, 0, 0),
            ivec3(0, -1, 0),
            ivec3(0, 1, 0),
            ivec3(0, 0, -1),
            ivec3(0, 0, 1),
        };
};

// implements n-dimensional wave function collapse with the "overlapping model"
// T: pattern data type
// N: number of dimension
// S: size of patterns (MUST BE ODD!)
// D: bitset size
// V (defaulted): ivecN
template <
    typename T,
    usize N,
    usize S,
    usize D,
    typename V = math::vec<N, int, math::defaultp>>
    requires ((S % 2) == 1)
struct WFC {
    // forward declarations
    struct Pattern;
    struct Element;
    struct Wave;

    // function which takes current wave and returns next cell to collapse
    using NextCellFn = std::function<Element&(Wave&)>;

    // function which chooses the pattern to collapse a cell to
    using PatternFn = std::function<usize(Wave&, Element&)>;

    // function used for optional callbacks during collapse process
    using CallbackFn = std::function<void(const Wave&)>;

    // S ** N (volume of one pattern)
    static constexpr auto VOL = math::cexp::pow(S, N);

    // pattern derived from input data
    struct Pattern {
        // unique pattern ID
        usize id;

        // normalized frequency of this pattern in input data
        f32 frequency;

        // value at the center of this pattern
        T value;

        // pattern data
        std::array<T, VOL> data;

        // valid neighbors on each side of this pattern
        std::array<Bitset<D>, 2 * N> valid;

        explicit Pattern(const std::array<T, VOL> &data)
            : data(data),
              value(ndarray::at(V(S), &data[0], V(S) / 2)) {}

        // NOTE: comparison is on data only
        auto operator<=>(const Pattern &other) const {
            return (*this) == other ?
                std::strong_ordering::equal
                : this->hash() <=> other.hash();
        }

        // NOTE: equality comparison is on data only
        bool operator==(const Pattern &other) const {
            return this->hash() == other.hash()
                && this->data == other.data;
        }

        inline u64 hash() const {
            if (this->_hash) {
                return this->_hash;
            }

            u64 v = 0x12345;
            for (const auto &x : this->data) {
                v ^= x + 0x9e3779b9 + (v << 6) + (v >> 2);
            }
            return this->_hash = v;
        }

private:
        // stored hash, only calculated once
        mutable u64 _hash = 0;
    };

    // wave element
    struct Element {
        // position in output space
        V pos;

        // coefficient, marked bits are VALID choices
        Bitset<D> c;

        // number of valid bits remaining
        usize popcnt;

        // value post-collapse (std::nullopt if not collapsed)
        std::optional<T> value = std::nullopt;

        // memoized entropy values
        f32 sum_weights = 0.0f;
        f32 sum_weight_log_weights = 0.0f;
        f32 entropy = 0.0f;

        // intialize entropy values, coefficient
        void init(const Wave &w, const Bitset<D> &mask) {
            this->c = mask;
            this->popcnt = this->c.popcnt();

            for (
                auto it = this->c.begin_on();
                it != this->c.end_on();
                it++) {
                const auto weight = w.wfc.patterns[*it].frequency;
                this->sum_weights += weight;
                this->sum_weight_log_weights += weight * std::log(weight);
            }
        }

        // applies a mask to this element's coefficient, updating memoized
        // entropy values
        // returns false on failure (0 popcnt/contradiction)
        bool apply(const Wave &w, const Bitset<D> &mask) {
            // get bits which are on in the current coefficient but off in the
            // new mask. exit early if nothing changes.
            const auto diff = this->c & mask;
            if (diff.popcnt() == 0) {
                return true;
            }

            this->c &= mask;

            for (
                auto it = diff.begin_on();
                it != diff.end_on();
                it++) {
                const auto weight = w.wfc.patterns[*it].frequency;
                this->sum_weights -= weight;
                this->sum_weight_log_weights -= weight * std::log(weight);
            }

            this->entropy =
                std::log(this->sum_weights)
                    - (this->sum_weight_log_weights / this->sum_weights);
            this->popcnt = this->c.popcnt();
            return this->popcnt != 0;
        }

        // collapse this element to pattern n
        bool collapse(usize n, const T &value) {
            if (!this->c[n]) {
                return false;
            }

            this->value = value;
            this->c.reset();
            this->c.set(n);
            this->popcnt = 1;
            this->entropy = 0.0f;
            this->sum_weights = 0.0f;
            this->sum_weight_log_weights = 0.0f;
            return true;
        }

        // returns true if this element is collapsed
        bool collapsed() const {
            return static_cast<bool>(this->value);
        }
    };

    // wave to collapse
    struct Wave {
        const WFC &wfc;

        // size of output wave
        V size_wave;

        // output wave elements
        std::vector<Element> wave;

        // optional preset values
        const std::optional<T> *preset;

        // total number of collapsed elements
        usize num_collapsed = 0;

        Wave(
            const WFC &wfc,
            const V &size_wave,
            const std::optional<T> *preset = nullptr)
            : wfc(wfc),
              size_wave(size_wave),
              preset(preset) {}

        // collapses the specified wave element to the only remaining
        // possibility, or n if specified
        result::Result<void, V> collapse(
            Element &e,
            usize n = std::numeric_limits<usize>::max()) {
            n = (n == std::numeric_limits<usize>::max()) ? e.c.nth_set(0) : n;

            const auto &p = this->wfc.patterns[n];
            if (!e.collapse(n, p.value)) {
                return result::Err(e.pos);
            }

            this->num_collapsed++;
            return result::Ok();
        }

        // observe specified wave element, collapsing it to one of its possible
        // values
        result::Result<void, V> observe(Element &e) {
            const auto n = this->wfc.pattern_fn(*this, e);
            return this->collapse(e, n);
        }

        // propagate the current value of the specified wave element
        // returns Ok on success, erroneous position V on failure
        // (contradiction)
        result::Result<void, V> propagate(Element &to_propagate) {
            // DFS elements to update
            std::stack<Element*> es;
            es.push(to_propagate);

            // if not std::nullopt, there is an unresolvable contradiction in
            // the wave
            std::optional<V> contradiction = std::nullopt;

            // propagate each stack entry to neighbors
            while (!es.empty()) {
                auto &e = *es.top();
                es.pop();

                // get only valid non-collapsed neighbors
                std::array<Element*, N * 2> neighbors;
                for (usize i = 0; i < N * 2; i++) {
                    neighbors[i] = nullptr;

                    const auto &n = Neighbors<N>::neighbors[i];
                    const auto pos_n = e.pos + n;

                    if (!ndarray::in_bounds(this->size_wave, pos_n)) {
                        continue;
                    }

                    auto &e_n =
                        ndarray::at(
                            this->size_wave,
                            &this->wave[0],
                            pos_n);

                    if (e_n.collapsed()) {
                        continue;
                    }

                    neighbors[i] = &e_n;
                }

                // compute superpatterns for valid neighbors
                std::array<Bitset<D>, N * 2> neighbor_patterns;
                for (
                    auto it = e.c.begin_on();
                    it != e.c.end_on();
                    it++) {
                    const auto &p = this->wfc.patterns[*it];
                    for (usize i = 0; i < N * 2; i++) {
                        if (neighbors[i]) {
                            neighbor_patterns[i] |= p.valid[i];
                        }
                    }
                }

                // apply superpatterns
                for (usize i = 0; i < N * 2; i++) {
                    if (!neighbors[i]) {
                        continue;
                    }

                    auto &e_n = *neighbors[i];
                    const auto popcnt_old = e_n.popcnt;
                    e_n.apply(*this, neighbor_patterns[i]);

                    if (e_n.popcnt != popcnt_old) {
                        if (e_n.popcnt == 0) {
                            // zero popcount = failure/contradiction
                            this->collapse(e_n, 0);
                            contradiction = e_n.pos;
                            break;
                        } else if (e_n.popcnt == 1) {
                            // one popcount = one remaining possibility/collapse
                            auto res = this->collapse(e_n);
                            if (res.isErr()) {
                                return res;
                            }
                        }

                        // propagate changed value
                        es.push(&e_n);
                    }
                }
            }

            if (contradiction) {
                return result::Err(*contradiction);
            }

            if (this->wfc.on_propagate) {
                (*this->wfc.on_propagate)(*this);
            }

            return result::Ok();
        }

        // collapse the wave
        result::Result<void, V> collapse() {
            // initialize the wave, all patterns are valid for each element
            this->wave = std::vector<Element>(math::prod(this->size_wave));

            // safe to parallelize, only one element modified at a time
            #pragma omp parallel for
            for (usize i = 0; i < this->wave.size(); i++) {
                auto &e = this->wave[i];
                e.pos = ndarray::unravel_index(this->size_wave, i);
                e.init(*this, this->wfc.mask_used);
            }

            // load preset values if present
            if (this->preset) {
                ndarray::each(
                    this->size_wave,
                    [&](const V &pos) {
                        const auto &p =
                            ndarray::at(this->size_wave, &this->preset[0], pos);
                        if (p) {
                            auto &e =
                                ndarray::at(
                                    this->size_wave,
                                    &this->wave[0],
                                    pos);
                            e.value = *p;
                        }
                    });
            }

            // collapse the wave
            while (this->num_collapsed != this->wave.size()) {
                auto &e = this->wfc.next_cell_fn(*this);

                const auto res_observe = this->observe(e);
                if (res_observe.isErr()) {
                    return res_observe;
                }

                const auto res_prop = this->propagate(e);
                if (res_prop.isErr()) {
                    return res_prop;
                }
            }

            // success!
            return result::Ok();
        }
    };

    // size of input data
    V size_in;

    // input data
    const T *in;

    // possible patterns
    std::vector<Pattern> patterns;

    // function to select pattern to collapse to
    PatternFn pattern_fn;

    // function to select next cell to collapse
    NextCellFn next_cell_fn;

    // behavior of patterns on borders
    BorderBehavior border_behavior;

    // mask of used bits of coefficient bitsets
    // bits are also zeroed for disallowed patterns (patterns without valid
    // neighbors)
    Bitset<D> mask_used;

    // random generator
    Rand rand;

    // options
    usize flags;

    // optional callback to be called after propagation of each collapsed wave
    // element
    std::optional<CallbackFn> on_propagate = std::nullopt;

    explicit WFC(
        const V &size_in,
        const T *in,
        PatternFunction pattern_function,
        NextCellFunction next_cell_function,
        BorderBehavior border_behavior,
        Rand &&rand,
        usize flags)
        : size_in(size_in),
          in(in),
          border_behavior(border_behavior),
          rand(std::move(rand)),
          flags(flags) {
        // get pattern data from input at specified location
        // returns std::nullopt if pattern is made illegal by border behavior
        const auto data_at =
            [&](const V &center) -> std::optional<std::array<T, VOL>> {
                const auto base = center - (V(S) / 2);

                if (border_behavior == BorderBehavior::EXCLUDE) {
                    if (!ndarray::in_bounds(size_in, base)
                            || !ndarray::in_bounds(size_in, base + V(S) - 1)) {
                        return std::nullopt;
                    }
                }

                std::array<T, VOL> dst;
                ndarray::each(
                    V(S),
                    [&](const V &offset) {
                        std::optional<T> override = std::nullopt;
                        V pos = base + offset;

                        switch (border_behavior) {
                            case BorderBehavior::EXCLUDE:
                                break;
                            case BorderBehavior::ZERO:
                                if (!ndarray::in_bounds(size_in, pos)) {
                                    override = T(0);
                                }
                                break;
                            case BorderBehavior::CLAMP:
                                pos = math::clamp(pos, V(0), size_in - V(1));
                                break;
                            case BorderBehavior::WRAP:
                                pos = ((pos % size_in) + size_in) % size_in;
                                break;
                        };

                        ndarray::at(V(S), &dst[0], offset) =
                            override ?
                                *override
                                : ndarray::at(size_in, in, pos);
                    });
                return std::make_optional(dst);
            };

        // slide along input data creating patterns
        ndarray::each(
            size_in,
            [&](const V &p) {
                if (const auto data = data_at(p)) {
                    this->patterns.emplace_back(Pattern(data));
                }
            });

        // calculate per-pattern frequencies (normalization occurs later once
        // patterns have been deduplicated)
        std::unordered_map<u64, usize> pattern_hash_to_freq;
        for (const auto &p : this->patterns) {
            pattern_hash_to_freq[p.hash()]++;
        }

        // assign base frequencies (again, not normalized)
        for (auto &p : this->patterns) {
            p.frequency = static_cast<f32>(pattern_hash_to_freq[p.hash()]);
        }

        // removes duplicate patterns from this->patterns
        const auto deduplicate =
            [&]() {
                std::sort(this->patterns.begin(), this->patterns.end());
                this->patterns.erase(
                    std::unique(this->patterns.begin(), this->patterns.end()),
                    this->patterns.end());
            };

        // remove duplicate patterns
        deduplicate();

        // add rotations, reflections
        const auto base_patterns = this->patterns;
        this->patterns.clear();
        for (const auto &p : base_patterns) {
            this->patterns.push_back(p);

            if (flags & FLAGS_ROTATE) {
                ASSERT(N == 2, "can only rotate patterns in 2 dimensions");
                if constexpr (N == 2) {
                    const auto permutations =
                        Rotator<N>::template permute<T, S>(p.data);
                    for (usize i = 1; i < permutations.size(); i++) {
                        auto &q =
                            this->patterns.emplace_back(
                                pattern(permutations[i]));
                        q.frequency = p.frequency;
                    }
                }
            }
            if (flags & FLAGS_REFLECT) {
                const auto permutations =
                    Reflector<N>::template permute<T, S>(p.data);
                for (usize i = 1; i < permutations.size(); i++) {
                    auto &q =
                        this->patterns.emplace_back(
                            pattern(permutations[i]));
                    q.frequency = p.frequency;
                }
            }
        }

        // deduplicate again to remove duplicates added by reflection/rotation
        deduplicate();

        // normalize pattern frequencies, pattern set will no longer change
        f32 frequency_total = 0.0f;
        for (const auto &p : this->patterns) {
            frequency_total += p.frequency;
        }

        for (auto &p : this->patterns) {
            p.frequency /= frequency_total;
        }

        // compute used mask, assign IDs
        this->mask_used.reset();
        for (usize i = 0; i < this->patterns.size(); i++) {
            this->patterns[i].id = i;
            this->mask_used.set(i);
        }

        // compuate valid patterns around each pattern (adjacency)
        // check against all overlapping slots for each pattern pair (p, q) for
        // every side
        //
        // safe to parallelize on p because only p is being written to and not
        // read from :)
        #pragma omp parallel for
        for (auto &p : this->patterns) {
            for (const auto &q : this->patterns) {
                for (usize i = 0; i < 2 * N; i++) {
                    const auto &n = Neighbors<N>::neighbors[i];
                    bool valid = true;
                    ndarray::each(
                        V(S),
                        [&](const auto &offset_q) {
                            if (!valid) {
                                return;
                            }

                            // compute offset into p's data
                            const auto offset_p = n + offset_q;
                            if (!ndarray::in_bounds(V(S), offset_p)) {
                                return;
                            }

                            const auto
                                v_p =
                                    ndarray::at(V(S), &p.data[0], offset_p),
                                v_q =
                                    ndarray::at(V(S), &q.data[0], offset_q);

                            // data must be equal at each offset for patterns to
                            // match (be valid neighbors)
                            if (v_p != v_q) {
                                valid = false;
                            }
                        });

                    if (valid) {
                        p.valid[i].set(q.id);
                    }
                }
            }
        }

        switch (pattern_function) {
            case PatternFunction::WEIGHTED:
                this->pattern_fn = this->pattern_weighted();
                break;
        }

        switch (next_cell_fn) {
            case NextCellFunction::MIN_ENTROPY:
                this->next_cell_fn = this->next_cell_min_entropy();
                break;
        }
    }

    // collapses a wave into the specified output
    // returns true on success
    bool collapse(
        const V &size_out,
        T *out,
        const std::optional<T> *preset = nullptr) const {
        auto w = Wave(*this, size_out, preset);
        const auto res = w.collapse();
        if (res.isErr()) {
            return false;
        }

        // map elements to output
        ndarray::each(
            w.size_wave,
            [&](const V &pos) {
                ndarray::at(size_out, out, pos) =
                    *ndarray::at(size_out, &w.wave, pos).value;
            });

        return true;
    }

    // returns a function which selects a pattern based on the distribution of
    // patterns in the input data
    PatternFn pattern_weighted() {
        return [&](Wave &w, Element &e) -> usize {
            f32 sum_cs = 0.0f;
            std::vector<f32> cs(this->patterns.size());
            for (
                auto it = e.c.begin_on();
                it != e.c.end_on();
                it++) {
                cs[*it] = this->patterns[*it].frequency;
                sum_cs += cs[*it];
            }

            const auto r = this->rand.template next<f32>(0.0f, sum_cs);
            f32 acc = 0.0f;
            for (usize i = 0; i < cs.size(); i++) {
                acc += cs[i];

                if (acc >= r) {
                    return i;
                }
            }

            ASSERT(false, "failed to select pattern for {}", e.pos);
            return e.c.nth_set(0);
        };
    }

    // returns a function which selects the next cell based on finding the cell
    // with the minimum entropy in those remaining
    NextCellFn next_cell_min_entropy() {
        return [&](Wave &w) -> Element& {
            f32 min = 1e4;
            Element *argmin = nullptr;

            // min entropy with a bit of noise
            for (auto &e : w.wave) {
                if (!e.collapsed()
                        && e.entropy < min
                        && e.entropy + rand.next<f32>(0.0f, 1e-6) < min) {
                    argmin = &e;
                }
            }

            ASSERT(argmin);
            return *argmin;
        };
    }
};
}
