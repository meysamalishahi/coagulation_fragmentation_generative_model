#!/usr/bin/env python3
"""CPU-only validation suite for the coagulation--fragmentation paper.

The script intentionally depends only on the Python standard library.  It
checks identities that can be validated without training a large neural
network or using a GPU:

* conservation and inversion of split/merge and birth/deletion maps;
* the closed-form evaporation flow and its Jacobian;
* the split and dequantization Jacobians by finite differences;
* the continuous split change-of-variables identity that produces 1/m;
* exact probability-flux time reversal on an enumerated finite state space;
* the uniform birth bridge and the survival-integral estimator;
* a small synthetic pair-structure diagnostic comparing independent births
  with a coagulation-aware parent/separation parameterization.

This is a correctness suite, not a substitute for the paper's real-data and
competitive-baseline experiments.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


Vector = tuple[float, ...]
Component = tuple[float, Vector]
State = tuple[float, tuple[Component, ...]]
FiniteComponent = tuple[int, int]
FiniteState = tuple[int, tuple[FiniteComponent, ...]]


@dataclass
class CheckResult:
    name: str
    passed: bool
    metric: float
    tolerance: float
    detail: str
    seconds: float


def _canonical(v: float, components: Iterable[Component]) -> State:
    return float(v), tuple(sorted((float(m), tuple(map(float, z))) for m, z in components))


def total_mass(state: State) -> float:
    return state[0] + sum(m for m, _ in state[1])


def weighted_feature(state: State) -> Vector:
    if not state[1]:
        return ()
    q = len(state[1][0][1])
    return tuple(sum(m * z[k] for m, z in state[1]) for k in range(q))


def split_component(state: State, index: int, rho: float, u: Vector) -> tuple[State, tuple[int, int]]:
    v, components = state
    m, z = components[index]
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must lie in (0,1)")
    if len(z) != len(u):
        raise ValueError("feature and displacement dimensions differ")
    ma, mb = rho * m, (1.0 - rho) * m
    za = tuple(zk - (1.0 - rho) * uk for zk, uk in zip(z, u))
    zb = tuple(zk + rho * uk for zk, uk in zip(z, u))
    tagged = [(component, old_index) for old_index, component in enumerate(components) if old_index != index]
    tagged.extend([((ma, za), -1), ((mb, zb), -2)])
    tagged.sort(key=lambda item: item[0])
    child_a = next(i for i, (_, tag) in enumerate(tagged) if tag == -1)
    child_b = next(i for i, (_, tag) in enumerate(tagged) if tag == -2)
    return _canonical(v, [component for component, _ in tagged]), (child_a, child_b)


def merge_components(state: State, a: int, b: int) -> State:
    if a == b:
        raise ValueError("a merge requires two distinct components")
    if a > b:
        a, b = b, a
    v, components = state
    ma, za = components[a]
    mb, zb = components[b]
    m = ma + mb
    z = tuple((ma * x + mb * y) / m for x, y in zip(za, zb))
    remaining = [c for i, c in enumerate(components) if i not in (a, b)]
    remaining.append((m, z))
    return _canonical(v, remaining)


def delete_component(state: State, index: int) -> tuple[State, Component]:
    v, components = state
    removed = components[index]
    remaining = [c for i, c in enumerate(components) if i != index]
    return _canonical(v + removed[0], remaining), removed


def birth_component(state: State, component: Component) -> State:
    v, components = state
    m, z = component
    if not (0.0 < m <= v + 1e-15):
        raise ValueError("birth mass must lie in (0,v]")
    return _canonical(v - m, list(components) + [(m, z)])


def flow_state(state: State, M: float, G: float) -> State:
    """Apply the forward closed-form flow with integrated rate G."""
    v, components = state
    c = sum(m for m, _ in components)
    if c == 0.0 or math.isclose(c, M, rel_tol=0.0, abs_tol=1e-15):
        return state
    e = math.exp(-G)
    c_new = M * c * e / (M - c + c * e)
    alpha = c_new / c
    return _canonical(M - c_new, [(alpha * m, z) for m, z in components])


def encode_weights(weights: Sequence[float], features: Sequence[Vector], M: float, xi: float) -> State:
    if len(weights) != len(features) or not weights:
        raise ValueError("weights and features must be nonempty and equally sized")
    if any(w <= 0.0 for w in weights) or abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("weights must be in the open simplex")
    if not 0.0 < xi < 1.0:
        raise ValueError("xi must lie in (0,1)")
    return _canonical(M * xi, [(M * (1.0 - xi) * w, z) for w, z in zip(weights, features)])


def decode_weights(state: State) -> list[float]:
    c = sum(m for m, _ in state[1])
    if c <= 0.0:
        raise ValueError("the empty state has no normalized weights")
    return [m / c for m, _ in state[1]]


def determinant(matrix: Sequence[Sequence[float]]) -> float:
    a = [list(map(float, row)) for row in matrix]
    n = len(a)
    sign = 1.0
    det = 1.0
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-15:
            return 0.0
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            sign *= -1.0
        value = a[col][col]
        det *= value
        for row in range(col + 1, n):
            factor = a[row][col] / value
            for k in range(col + 1, n):
                a[row][k] -= factor * a[col][k]
    return sign * det


def numerical_jacobian(function: Callable[[list[float]], list[float]], x: list[float], h: float = 1e-6) -> list[list[float]]:
    y0 = function(x)
    jac = [[0.0 for _ in x] for _ in y0]
    for j in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        yp, ym = function(xp), function(xm)
        for i in range(len(y0)):
            jac[i][j] = (yp[i] - ym[i]) / (2.0 * h)
    return jac


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-15)


def timed_check(name: str, tolerance: float, detail: str, function: Callable[[], float]) -> CheckResult:
    start = time.perf_counter()
    try:
        metric = float(function())
        passed = math.isfinite(metric) and metric <= tolerance
        suffix = detail
    except Exception as exc:  # A failed check should not hide subsequent checks.
        metric = math.inf
        passed = False
        suffix = f"{detail}; exception={type(exc).__name__}: {exc}"
    return CheckResult(name, passed, metric, tolerance, suffix, time.perf_counter() - start)


def check_event_maps(rng: random.Random, trials: int = 2000) -> float:
    worst = 0.0
    for _ in range(trials):
        q = 2
        masses = [rng.uniform(0.2, 3.0) for _ in range(3)]
        v = rng.uniform(0.2, 3.0)
        state = _canonical(v, [(m, (rng.gauss(0, 1), rng.gauss(0, 1))) for m in masses])
        M = total_mass(state)
        index = rng.randrange(len(state[1]))
        rho = rng.uniform(0.05, 0.95)
        u = tuple(rng.uniform(0.1, 2.0) for _ in range(q))
        before_feature = weighted_feature(state)
        split, children = split_component(state, index, rho, u)
        recovered = merge_components(split, *children)
        err_mass = abs(total_mass(split) - M) / M
        err_feature = max(abs(a - b) for a, b in zip(before_feature, weighted_feature(split))) / max(
            1.0, max(map(abs, before_feature))
        )
        # Compare canonical coordinate lists directly after the exact inverse.
        coord_err = max(
            [abs(recovered[0] - state[0])]
            + [abs(a[0] - b[0]) for a, b in zip(recovered[1], state[1])]
            + [abs(x - y) for a, b in zip(recovered[1], state[1]) for x, y in zip(a[1], b[1])]
        )
        deleted, component = delete_component(state, index)
        reborn = birth_component(deleted, component)
        birth_err = max(
            [abs(reborn[0] - state[0])]
            + [abs(a[0] - b[0]) for a, b in zip(reborn[1], state[1])]
            + [abs(x - y) for a, b in zip(reborn[1], state[1]) for x, y in zip(a[1], b[1])]
        )
        worst = max(worst, err_mass, err_feature, coord_err, birth_err)
    return worst


def check_flow(rng: random.Random, trials: int = 1000) -> float:
    worst = 0.0
    for _ in range(trials):
        M = rng.uniform(2.0, 20.0)
        raw = [rng.expovariate(1.0) for _ in range(4)]
        scale = rng.uniform(0.05, 0.95) * M / sum(raw)
        state = _canonical(M - scale * sum(raw), [(scale * m, (rng.gauss(0, 1),)) for m in raw])
        G = rng.uniform(0.0, 3.0)
        forward = flow_state(state, M, G)
        recovered = flow_state(forward, M, -G)
        error = max(
            abs(total_mass(forward) - M) / M,
            abs(recovered[0] - state[0]) / M,
            max(abs(a[0] - b[0]) for a, b in zip(recovered[1], state[1])) / M,
        )
        worst = max(worst, error)
    return worst


def check_encoder_decoder(rng: random.Random, trials: int = 1000) -> float:
    worst = 0.0
    for _ in range(trials):
        n = rng.randint(1, 8)
        raw = [rng.expovariate(1.0) for _ in range(n)]
        weights = [x / sum(raw) for x in raw]
        features = [(rng.gauss(0, 1),) for _ in range(n)]
        M, xi = rng.uniform(0.5, 20.0), rng.uniform(0.01, 0.99)
        state = encode_weights(weights, features, M, xi)
        decoded_by_feature = {z: w for w, (_, z) in zip(decode_weights(state), state[1])}
        original_by_feature = {z: w for w, z in zip(weights, features)}
        worst = max(worst, max(abs(decoded_by_feature[z] - w) for z, w in original_by_feature.items()))
    return worst


def check_split_jacobian(rng: random.Random, trials: int = 300) -> float:
    worst = 0.0
    for _ in range(trials):
        q = rng.choice((1, 2, 3))
        m, rho = rng.uniform(0.3, 4.0), rng.uniform(0.1, 0.9)
        z = [rng.uniform(-1.0, 1.0) for _ in range(q)]
        u = [rng.uniform(0.1, 1.0) for _ in range(q)]
        x = [m, rho] + z + u

        def mapping(values: list[float]) -> list[float]:
            mm, rr = values[:2]
            zz, uu = values[2 : 2 + q], values[2 + q :]
            ma, mb = rr * mm, (1.0 - rr) * mm
            za = [a - (1.0 - rr) * b for a, b in zip(zz, uu)]
            zb = [a + rr * b for a, b in zip(zz, uu)]
            return [ma, mb] + za + zb

        observed = abs(determinant(numerical_jacobian(mapping, x)))
        worst = max(worst, relative_error(observed, m))
    return worst


def check_dequantization_jacobian(rng: random.Random, trials: int = 300) -> float:
    worst = 0.0
    for _ in range(trials):
        n = rng.randint(1, 7)
        raw = [rng.expovariate(1.0) for _ in range(n)]
        weights = [x / sum(raw) for x in raw]
        M, xi = rng.uniform(0.5, 10.0), rng.uniform(0.1, 0.9)
        x = weights[:-1] + [xi]

        def mapping(values: list[float]) -> list[float]:
            free, xx = values[:-1], values[-1]
            ww = free + [1.0 - sum(free)]
            return [M * (1.0 - xx) * w for w in ww]

        observed = abs(determinant(numerical_jacobian(mapping, x, h=2e-7)))
        expected = M**n * (1.0 - xi) ** (n - 1)
        worst = max(worst, relative_error(observed, expected))
    return worst


def check_flow_jacobian(rng: random.Random, trials: int = 300) -> float:
    worst = 0.0
    for _ in range(trials):
        n = rng.randint(1, 7)
        M = rng.uniform(2.0, 12.0)
        raw = [rng.expovariate(1.0) for _ in range(n)]
        scale = rng.uniform(0.05, 0.9) * M / sum(raw)
        masses = [scale * x for x in raw]
        c, G = sum(masses), rng.uniform(0.0, 2.0)

        def mapping(values: list[float]) -> list[float]:
            cc = sum(values)
            e = math.exp(-G)
            f = M * cc * e / (M - cc + cc * e)
            return [f / cc * x for x in values]

        observed = abs(determinant(numerical_jacobian(mapping, masses)))
        e = math.exp(-G)
        f = M * c * e / (M - c + c * e)
        alpha = f / c
        derivative = M * M * e / (M - c + c * e) ** 2
        expected = alpha ** (n - 1) * derivative
        worst = max(worst, relative_error(observed, expected))
    return worst


def check_split_change_of_variables(rng: random.Random, samples: int = 500_000) -> float:
    """Independent Monte Carlo integrals on the two sides of dy=m dx.

    Source domain: m in [.5,2], rho in [.2,.8], z in [-1,1], u in [.1,1].
    Target samples cover a box containing the complete image; an inverse-map
    indicator selects the image.  This does not reuse the source samples.
    """
    source_volume = 1.5 * 0.6 * 2.0 * 0.9
    target_ranges = ((0.1, 1.6), (0.1, 1.6), (-1.8, 0.92), (-0.92, 1.8))
    target_volume = math.prod(b - a for a, b in target_ranges)

    def test_function(ma: float, mb: float, za: float, zb: float) -> float:
        return math.exp(-0.25 * (ma * ma + mb * mb + za * za + zb * zb))

    source_sum = 0.0
    for _ in range(samples):
        m = rng.uniform(0.5, 2.0)
        rho = rng.uniform(0.2, 0.8)
        z = rng.uniform(-1.0, 1.0)
        u = rng.uniform(0.1, 1.0)
        ma, mb = rho * m, (1.0 - rho) * m
        za, zb = z - (1.0 - rho) * u, z + rho * u
        source_sum += test_function(ma, mb, za, zb) * m
    source_integral = source_volume * source_sum / samples

    target_sum = 0.0
    for _ in range(samples):
        ma, mb, za, zb = [rng.uniform(a, b) for a, b in target_ranges]
        m = ma + mb
        rho = ma / m
        u = zb - za
        z = (ma * za + mb * zb) / m
        inside = 0.5 <= m <= 2.0 and 0.2 <= rho <= 0.8 and -1.0 <= z <= 1.0 and 0.1 <= u <= 1.0
        if inside:
            target_sum += test_function(ma, mb, za, zb)
    target_integral = target_volume * target_sum / samples
    return relative_error(target_integral, source_integral)


def enumerate_finite_states(M: int, K: int, nmax: int) -> list[FiniteState]:
    component_types = [(m, z) for m in range(1, M + 1) for z in range(1, K + 1)]
    states: list[FiniteState] = [(M, ())]
    for n in range(1, nmax + 1):
        for components in itertools.combinations_with_replacement(component_types, n):
            condensed = sum(m for m, _ in components)
            if condensed <= M:
                states.append((M - condensed, tuple(components)))
    return states


def finite_split_marks(component: FiniteComponent, K: int) -> list[tuple[FiniteComponent, FiniteComponent]]:
    m, z = component
    marks = []
    for ma in range(1, m):
        mb = m - ma
        for za in range(1, K + 1):
            for zb in range(za + 1, K + 1):
                if ma * za + mb * zb == m * z:
                    marks.append(((ma, za), (mb, zb)))
    return marks


def finite_transitions(
    state: FiniteState,
    index: dict[FiniteState, int],
    K: int,
    nmax: int,
    t: float,
    tau: float,
    T: float,
    beta: float,
) -> dict[int, float]:
    v, components = state
    rates: dict[int, float] = defaultdict(float)
    if t <= tau and len(components) < nmax:
        for i, component in enumerate(components):
            marks = finite_split_marks(component, K)
            if not marks:
                continue
            for child_a, child_b in marks:
                new_components = list(components)
                new_components.pop(i)
                new_components.extend((child_a, child_b))
                target = (v, tuple(sorted(new_components)))
                rates[index[target]] += beta / len(marks)
    elif t > tau:
        delta = 1.0 / (T - t)
        for i, component in enumerate(components):
            new_components = list(components)
            new_components.pop(i)
            target = (v + component[0], tuple(new_components))
            rates[index[target]] += delta
    return dict(rates)


def generator_action(
    probability: Sequence[float],
    states: Sequence[FiniteState],
    index: dict[FiniteState, int],
    K: int,
    nmax: int,
    t: float,
    tau: float,
    T: float,
    beta: float,
) -> list[float]:
    derivative = [0.0] * len(states)
    for i, mass in enumerate(probability):
        if mass == 0.0:
            continue
        transitions = finite_transitions(states[i], index, K, nmax, t, tau, T, beta)
        outgoing = sum(transitions.values())
        derivative[i] -= mass * outgoing
        for j, rate in transitions.items():
            derivative[j] += mass * rate
    return derivative


def add_scaled(base: Sequence[float], increment: Sequence[float], scale: float) -> list[float]:
    return [a + scale * b for a, b in zip(base, increment)]


def integrate_forward(
    initial: Sequence[float],
    states: Sequence[FiniteState],
    index: dict[FiniteState, int],
    K: int,
    nmax: int,
    tau: float,
    T: float,
    beta: float,
    targets: Sequence[float],
    max_step: float = 0.001,
) -> dict[float, list[float]]:
    p = list(initial)
    t = 0.0
    output: dict[float, list[float]] = {}
    for target in sorted(targets):
        while t < target - 1e-15:
            # Do not let an RK step straddle the schedule discontinuity.
            boundary = tau if t < tau < target else target
            dt = min(max_step, boundary - t)
            if dt <= 1e-15:
                t = boundary
                continue
            action = lambda vector, when: generator_action(vector, states, index, K, nmax, when, tau, T, beta)
            k1 = action(p, t)
            k2 = action(add_scaled(p, k1, dt / 2.0), t + dt / 2.0)
            k3 = action(add_scaled(p, k2, dt / 2.0), t + dt / 2.0)
            k4 = action(add_scaled(p, k3, dt), t + dt)
            p = [max(0.0, value + dt * (a + 2 * b + 2 * c + d) / 6.0) for value, a, b, c, d in zip(p, k1, k2, k3, k4)]
            normalization = sum(p)
            p = [value / normalization for value in p]
            t += dt
        output[target] = p.copy()
    return output


def finite_reverse_metrics() -> tuple[float, dict[str, float]]:
    M, K, nmax = 6, 3, 4
    tau, T, beta = 0.45, 1.0, 1.3
    states = enumerate_finite_states(M, K, nmax)
    index = {state: i for i, state in enumerate(states)}
    initial_state: FiniteState = (2, ((2, 2), (2, 2)))
    initial = [0.0] * len(states)
    initial[index[initial_state]] = 1.0
    times = (0.25, 0.75, 0.95)
    marginals = integrate_forward(initial, states, index, K, nmax, tau, T, beta, times)
    worst_flux = 0.0
    worst_derivative = 0.0
    worst_row_sum = 0.0
    positive_reverse_edges = 0
    for t in times[:-1]:
        p = marginals[t]
        forward_derivative = generator_action(p, states, index, K, nmax, t, tau, T, beta)
        reverse_rows: dict[int, dict[int, float]] = defaultdict(dict)
        for x, px in enumerate(p):
            if px <= 1e-14:
                continue
            for y, rate in finite_transitions(states[x], index, K, nmax, t, tau, T, beta).items():
                py = p[y]
                if py <= 1e-14:
                    continue
                reverse = rate * px / py
                reverse_rows[y][x] = reverse_rows[y].get(x, 0.0) + reverse
                flux_forward = px * rate
                flux_reverse = py * reverse
                worst_flux = max(worst_flux, abs(flux_forward - flux_reverse))
                positive_reverse_edges += 1
        reverse_derivative = [0.0] * len(states)
        for y, row in reverse_rows.items():
            outgoing = sum(row.values())
            row_sum = -outgoing + sum(row.values())
            worst_row_sum = max(worst_row_sum, abs(row_sum))
            reverse_derivative[y] -= p[y] * outgoing
            for x, rate in row.items():
                reverse_derivative[x] += p[y] * rate
        worst_derivative = max(worst_derivative, max(abs(a + b) for a, b in zip(reverse_derivative, forward_derivative)))
    terminal_empty_probability = marginals[0.95][index[(M, ())]]
    metric = max(worst_flux, worst_derivative, worst_row_sum)
    return metric, {
        "state_count": float(len(states)),
        "positive_reverse_edges": float(positive_reverse_edges),
        "worst_flux_residual": worst_flux,
        "worst_reverse_derivative_residual": worst_derivative,
        "worst_reverse_row_sum": worst_row_sum,
        "empty_probability_t_0.95": terminal_empty_probability,
    }


def finite_reverse_rows_at(t: float) -> tuple[list[FiniteState], dict[int, dict[int, float]]]:
    """Construct analytical reverse rows for the paper's finite benchmark."""
    M, K, nmax = 6, 3, 4
    tau, T, beta = 0.45, 1.0, 1.3
    states = enumerate_finite_states(M, K, nmax)
    index = {state: i for i, state in enumerate(states)}
    initial = [0.0] * len(states)
    initial[index[(2, ((2, 2), (2, 2)))]] = 1.0
    p = integrate_forward(initial, states, index, K, nmax, tau, T, beta, (t,))[t]
    reverse_rows: dict[int, dict[int, float]] = defaultdict(dict)
    for x, px in enumerate(p):
        if px <= 1e-14:
            continue
        for y, rate in finite_transitions(states[x], index, K, nmax, t, tau, T, beta).items():
            if p[y] <= 1e-14:
                continue
            reverse_rows[y][x] = reverse_rows[y].get(x, 0.0) + rate * px / p[y]
    return states, dict(reverse_rows)


def check_finite_rate_mle(rng: random.Random, repetitions_per_row: int = 80_000) -> float:
    """Recover frozen analytical reverse rates by tabular point-process MLE.

    Each reverse source state is observed for a censoring window 1/lambda.
    Event counts divided by total exposure are the exact maximum-likelihood
    estimates for the row's marked Poisson intensities.
    """
    total_absolute_error = 0.0
    total_rate = 0.0
    for t in (0.25, 0.75):
        _, rows = finite_reverse_rows_at(t)
        for rates in rows.values():
            total = sum(rates.values())
            if total <= 0.0:
                continue
            horizon = 1.0 / total
            destinations = list(rates)
            cumulative = []
            running = 0.0
            for destination in destinations:
                running += rates[destination]
                cumulative.append(running)
            counts = {destination: 0 for destination in destinations}
            exposure = 0.0
            for _ in range(repetitions_per_row):
                waiting = rng.expovariate(total)
                exposure += min(waiting, horizon)
                if waiting < horizon:
                    draw = rng.random() * total
                    selected = destinations[-1]
                    for destination, threshold in zip(destinations, cumulative):
                        if draw <= threshold:
                            selected = destination
                            break
                    counts[selected] += 1
            for destination, exact in rates.items():
                estimate = counts[destination] / exposure
                total_absolute_error += abs(estimate - exact)
                total_rate += exact
    return total_absolute_error / total_rate


def check_birth_bridge(rng: random.Random, repetitions: int = 60_000) -> float:
    S = 1.7
    worst = 0.0
    for L in range(1, 7):
        sums = [0.0] * L
        for _ in range(repetitions // L):
            ordered = sorted(rng.uniform(0.0, S) for _ in range(L))
            for k, value in enumerate(ordered):
                sums[k] += value
        count = repetitions // L
        for k, value in enumerate(sums, start=1):
            empirical = value / count
            exact = S * k / (L + 1)
            worst = max(worst, abs(empirical - exact) / S)
    return worst


def check_survival_integral(rng: random.Random, samples: int = 300_000) -> float:
    # Integral_0^2 (1 + s^2) ds = 14/3.
    estimate = 2.0 * sum(1.0 + (2.0 * rng.random()) ** 2 for _ in range(samples)) / samples
    return relative_error(estimate, 14.0 / 3.0)


def check_constant_rate_likelihood(rng: random.Random, samples: int = 200_000) -> float:
    rate = 2.3
    # For uncensored Exp(rate), E[-log(rate)+rate*T] = 1-log(rate).
    empirical = sum(-math.log(rate) + rate * rng.expovariate(rate) for _ in range(samples)) / samples
    expected = 1.0 - math.log(rate)
    return abs(empirical - expected)


def check_constant_rate_thinning(rng: random.Random, samples: int = 250_000) -> float:
    rate, envelope = 1.7, 3.2
    total_wait = 0.0
    for _ in range(samples):
        elapsed = 0.0
        while True:
            elapsed += rng.expovariate(envelope)
            if rng.random() < rate / envelope:
                total_wait += elapsed
                break
    empirical_mean = total_wait / samples
    return relative_error(empirical_mean, 1.0 / rate)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def covariance(x: Sequence[float], y: Sequence[float]) -> float:
    mx, my = mean(x), mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) - 1)


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def wasserstein_1d(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("equal sample sizes are required")
    return mean([abs(a - b) for a, b in zip(sorted(x), sorted(y))])


def synthetic_pair_diagnostic(rng: random.Random, train_size: int = 10_000, test_size: int = 20_000) -> dict[str, float]:
    """A small diagnostic where two children share a latent parent center.

    The birth-only sampler draws the two positions independently from their
    pooled marginal.  The structured sampler learns distributions of the
    conserved parent center and positive separation, then reconstructs the
    two children.  Empirical resampling is used so no numerical package is
    required.  This tests representational bias, not neural optimization.
    """

    def data_pair() -> tuple[float, float]:
        center = rng.gauss(0.0, 1.0)
        separation = math.exp(rng.gauss(-1.0, 0.22))
        return center - separation / 2.0, center + separation / 2.0

    train = [data_pair() for _ in range(train_size)]
    truth = [data_pair() for _ in range(test_size)]
    pooled = [z for pair in train for z in pair]
    centers = [(a + b) / 2.0 for a, b in train]
    separations = [b - a for a, b in train]

    independent = []
    structured = []
    for _ in range(test_size):
        independent.append(tuple(sorted((rng.choice(pooled), rng.choice(pooled)))))
        c, u = rng.choice(centers), rng.choice(separations)
        structured.append((c - u / 2.0, c + u / 2.0))

    def summaries(pairs: Sequence[tuple[float, float]]) -> tuple[list[float], list[float], list[float]]:
        left = [a for a, _ in pairs]
        right = [b for _, b in pairs]
        distances = [b - a for a, b in pairs]
        return left, right, distances

    truth_left, truth_right, truth_distance = summaries(truth)
    ind_left, ind_right, ind_distance = summaries(independent)
    str_left, str_right, str_distance = summaries(structured)
    truth_cov = covariance(truth_left, truth_right)
    ind_cov = covariance(ind_left, ind_right)
    str_cov = covariance(str_left, str_right)
    return {
        "truth_pair_covariance": truth_cov,
        "birth_only_pair_covariance": ind_cov,
        "structured_pair_covariance": str_cov,
        "birth_only_covariance_error": abs(ind_cov - truth_cov),
        "structured_covariance_error": abs(str_cov - truth_cov),
        "birth_only_distance_w1": wasserstein_1d(truth_distance, ind_distance),
        "structured_distance_w1": wasserstein_1d(truth_distance, str_distance),
        "truth_distance_median": quantile(truth_distance, 0.5),
        "birth_only_distance_median": quantile(ind_distance, 0.5),
        "structured_distance_median": quantile(str_distance, 0.5),
    }


def run_suite(seed: int, quick: bool) -> tuple[list[CheckResult], dict[str, object]]:
    rng = random.Random(seed)
    checks: list[CheckResult] = []
    checks.append(timed_check("event_maps_and_conservation", 2e-12, "split/merge and delete/birth inversion", lambda: check_event_maps(rng, 400 if quick else 2000)))
    checks.append(timed_check("closed_form_flow", 2e-12, "flow conservation and inverse flow", lambda: check_flow(rng, 250 if quick else 1000)))
    checks.append(timed_check("normalized_encoder_decoder", 2e-12, "raw weights recovered after reservoir augmentation", lambda: check_encoder_decoder(rng, 250 if quick else 1000)))
    checks.append(timed_check("split_jacobian", 2e-7, "finite-difference determinant versus parent mass m", lambda: check_split_jacobian(rng, 80 if quick else 300)))
    checks.append(timed_check("dequantization_jacobian", 1e-6, "finite-difference determinant versus M^n(1-xi)^(n-1)", lambda: check_dequantization_jacobian(rng, 80 if quick else 300)))
    checks.append(timed_check("flow_jacobian", 1e-6, "finite-difference determinant versus alpha^(n-1) f'(c)", lambda: check_flow_jacobian(rng, 80 if quick else 300)))
    checks.append(timed_check("split_change_of_variables", 0.018 if quick else 0.008, "independent Monte Carlo integrals; validates dy=m dx and reverse 1/m", lambda: check_split_change_of_variables(rng, 80_000 if quick else 500_000)))
    finite_start = time.perf_counter()
    try:
        finite_metric, finite_detail = finite_reverse_metrics()
        checks.append(CheckResult("finite_exact_time_reversal", finite_metric <= 2e-10, finite_metric, 2e-10, "flux equality and q R = -p G", time.perf_counter() - finite_start))
    except Exception as exc:
        finite_detail = {"exception": f"{type(exc).__name__}: {exc}"}
        checks.append(CheckResult("finite_exact_time_reversal", False, math.inf, 2e-10, "flux equality and q R = -p G", time.perf_counter() - finite_start))
    checks.append(timed_check("finite_reverse_rate_mle", 0.025 if quick else 0.012, "tabular marked-rate MLE versus analytical reverse rates", lambda: check_finite_rate_mle(rng, 20_000 if quick else 80_000)))
    checks.append(timed_check("uniform_birth_bridge", 0.005 if quick else 0.0025, "order-statistic means k S/(L+1)", lambda: check_birth_bridge(rng, 60_000 if quick else 600_000)))
    checks.append(timed_check("survival_integral_estimator", 0.006 if quick else 0.0025, "uniform-time Monte Carlo versus exact integral", lambda: check_survival_integral(rng, 60_000 if quick else 300_000)))
    checks.append(timed_check("constant_rate_path_likelihood", 0.012 if quick else 0.004, "mean exponential-event NLL versus analytic expectation", lambda: check_constant_rate_likelihood(rng, 100_000 if quick else 1_000_000)))
    checks.append(timed_check("constant_rate_thinning", 0.012 if quick else 0.005, "accepted waiting-time mean versus Exp(rate)", lambda: check_constant_rate_thinning(rng, 60_000 if quick else 250_000)))
    diagnostic_start = time.perf_counter()
    diagnostic = synthetic_pair_diagnostic(rng, 2500 if quick else 10_000, 5000 if quick else 20_000)
    diagnostic["structured_improves_covariance"] = diagnostic["structured_covariance_error"] < diagnostic["birth_only_covariance_error"]
    diagnostic["structured_improves_distance_w1"] = diagnostic["structured_distance_w1"] < diagnostic["birth_only_distance_w1"]
    diagnostic["seconds"] = time.perf_counter() - diagnostic_start
    metadata: dict[str, object] = {
        "seed": seed,
        "quick": quick,
        "finite_state": finite_detail,
        "synthetic_pair_diagnostic": diagnostic,
        "interpretation": {
            "validated": "algebraic maps, conservation, coordinate Jacobians, Monte Carlo 1/m identity, finite-state flux reversal, and simple point-process identities",
            "not_validated": "neural estimation, real-data sample quality, competitive baselines, scalability, or ICML-level empirical advantage",
        },
    }
    return checks, metadata


def write_outputs(output_dir: Path, checks: Sequence[CheckResult], metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "all_checks_passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
        **metadata,
    }
    (output_dir / "cpu_validation_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "cpu_validation_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(checks[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(check) for check in checks)


def print_report(checks: Sequence[CheckResult], metadata: dict[str, object]) -> None:
    print("CPU validation of the coagulation--fragmentation construction")
    print("=" * 68)
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{status:4s}  {check.name:34s} metric={check.metric:.6g}  tol={check.tolerance:.3g}  ({check.seconds:.2f}s)")
    finite = metadata["finite_state"]
    diagnostic = metadata["synthetic_pair_diagnostic"]
    print("\nFinite benchmark:")
    print(f"  enumerated states: {int(finite.get('state_count', 0))}")
    print(f"  tested positive reverse edges: {int(finite.get('positive_reverse_edges', 0))}")
    print(f"  P(empty at t=0.95): {finite.get('empty_probability_t_0.95', float('nan')):.6f}")
    print("\nSynthetic pair diagnostic (not a publication experiment):")
    print(f"  distance W1, independent birth: {diagnostic['birth_only_distance_w1']:.6f}")
    print(f"  distance W1, structured pair:   {diagnostic['structured_distance_w1']:.6f}")
    print(f"  covariance error, independent:  {diagnostic['birth_only_covariance_error']:.6f}")
    print(f"  covariance error, structured:   {diagnostic['structured_covariance_error']:.6f}")
    print("\nOverall:", "PASS" if all(check.passed for check in checks) else "FAIL")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--quick", action="store_true", help="run a shorter smoke test")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cpu_validation"))
    args = parser.parse_args()
    checks, metadata = run_suite(args.seed, args.quick)
    write_outputs(args.output_dir, checks, metadata)
    print_report(checks, metadata)
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
