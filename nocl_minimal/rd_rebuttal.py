"""NOCL Reaction--Diffusion from the recorded rebuttal experiment.

Numerical kernels, RNG roles, curriculum and evaluation follow the original
reaction_diffusion_paper_equation_nu_global_lr_dirichlet_v2 protocol.
This module adds portable checkpoints and a common experiment interface.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.fft import dst, idst

from .models import FNO2D
from .io import environment, save_params, write_json

PROTOCOL_ID = "reaction_diffusion_paper_equation_nu_global_lr_dirichlet_v2"

X_SIZE = 128
T_SIZE = 128
X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
TARGET_NU = 4.0
FIXED_RHO = 5.0
BOUNDARY_CONDITION = "u(0,t)=u(1,t)=0"

INITIAL_SUPPORT_COUNT = 10
CURRICULUM_STAGES = ((1.0, 2.0, 5), (2.0, 3.0, 5), (3.0, 4.0, 5))
CURRICULUM_TARGET_COUNT = sum(stage[2] for stage in CURRICULUM_STAGES)
TOTAL_BLOCKS = INITIAL_SUPPORT_COUNT + CURRICULUM_TARGET_COUNT

FULL_PHYSICS_STEPS_PER_BLOCK = 50_000
FULL_TOTAL_PHYSICAL_STEPS = TOTAL_BLOCKS * FULL_PHYSICS_STEPS_PER_BLOCK
FULL_WARMUP_STEPS = 10_000
FULL_FNO_STEPS = 3_000

PHYSICS_INIT_LR = 1e-3
PHYSICS_DECAY_RATE = 0.95
PHYSICS_TRANSITION_STEPS = 500
WARMUP_INIT_LR = 1e-3
WARMUP_DECAY_RATE = 0.9
WARMUP_TRANSITION_STEPS = 2_500
FNO_INIT_LR = 1e-3
FNO_DECAY_RATE = 0.95
FNO_TRANSITION_STEPS = 100

NUM_RESIDUAL_POINTS = 256
NUM_INITIAL_POINTS = 100
NUM_BOUNDARY_POINTS = 100
MASK_RATIO = 0.5
MASK_KEEP_POINTS = int(X_SIZE * T_SIZE * MASK_RATIO)

FULL_EVALUATION_SIZE = 1_280
SMOKE_EVALUATION_SIZE = 128
EVALUATION_CHUNK_SIZE = 65_536

SMOKE_PHYSICS_STEPS_PER_BLOCK = 2
SMOKE_WARMUP_STEPS = 2
SMOKE_FNO_STEPS = 2

RNG_ROLE_MLP_INITIALIZATION = 1
RNG_ROLE_PHYSICS_SAMPLING = 2
RNG_ROLE_FNO_INITIALIZATION = 3
RNG_ROLE_FNO_BATCH_SAMPLING = 4
RNG_ROLE_COMPILE_ONLY = 5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float(value: Any) -> float:
    return float(jax.device_get(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, jax.Array):
        array = np.asarray(jax.device_get(value))
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sync_tree(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def initial_nus() -> list[float]:
    return [float(value) for value in np.linspace(0.0, 1.0, 10)]


def stage_nus(low: float, high: float, count: int) -> list[float]:
    return [float(value) for value in np.linspace(low, high, count + 1)[1:]]


def curriculum_nus() -> list[float]:
    values = initial_nus()
    for low, high, count in CURRICULUM_STAGES:
        values.extend(stage_nus(low, high, count))
    if len(values) != TOTAL_BLOCKS:
        raise AssertionError(f"Expected {TOTAL_BLOCKS} nu values, got {len(values)}")
    if not np.isclose(values[-1], TARGET_NU):
        raise AssertionError("The nu curriculum must terminate at nu=4")
    return values



@dataclass(frozen=True)
class ResolvedConfig:
    seed: int
    smoke: bool
    physics_steps_per_block: int
    warmup_steps_per_target: int
    fno_steps_per_stage: int
    evaluation_size: int


def resolve_config(config):
    expected = {"schema_version", "problem", "rd_protocol", "seed", "smoke",
                "physics_steps_per_block", "warmup_steps_per_target",
                "fno_steps_per_stage", "evaluation_size"}
    extra = set(config) - expected
    if extra:
        raise ValueError(f"Unknown RD configuration fields: {sorted(extra)}")
    if config["rd_protocol"] != PROTOCOL_ID:
        raise ValueError("RD requires the recorded rebuttal protocol")
    smoke = bool(config.get("smoke", False))
    values = {key: config[key] for key in (
        "seed", "physics_steps_per_block", "warmup_steps_per_target",
        "fno_steps_per_stage", "evaluation_size")}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        if value < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}: {value}")
    if values["evaluation_size"] < 4:
        raise ValueError("evaluation_size must be at least 4")
    if smoke:
        values.update(physics_steps_per_block=2, warmup_steps_per_target=2,
                      fno_steps_per_stage=2, evaluation_size=128)
    return ResolvedConfig(smoke=smoke, **values)


class JsonRecorder:
    def __init__(
        self,
        run_dir: Path,
        protocol: dict[str, Any],
        metadata: dict[str, Any],
        overwrite: bool,
    ) -> None:
        self.run_dir = run_dir
        self.progress_path = run_dir / "progress.json"
        self.final_path = run_dir / "final.json"
        if self.final_path.exists() and not overwrite:
            raise FileExistsError(
                f"{self.final_path} already exists; pass --overwrite to rerun"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        self.payload: dict[str, Any] = {
            "status": "running",
            "started_at": _utc_now(),
            "protocol": protocol,
            "metadata": metadata,
            "timing": {
                "compile_seconds": 0.0,
                "pinn_parameter_initialization_seconds": 0.0,
                "physics_optimizer_initialization_seconds": 0.0,
                "initial_support_physics_seconds": 0.0,
                "curriculum_or_target_physics_seconds": 0.0,
                "fno_training_seconds": 0.0,
                "fno_inference_seconds": 0.0,
                "mask_seconds": 0.0,
                "supervised_warmup_seconds": 0.0,
                "operator_target_generation_seconds": 0.0,
                "evaluation_seconds": 0.0,
                "active_training_seconds": 0.0,
                "end_to_end_seconds": 0.0,
            },
            "counts": {
                "physical_steps": 0,
                "fno_steps": 0,
                "warmup_steps": 0,
            },
            "timing_details": {
                "reference_generation_seconds": 0.0,
                "grid_generation_seconds": 0.0,
                "physical_blocks": [],
                "fno_stages": [],
                "fno_inference": [],
                "masks": [],
                "warmups": [],
                "evaluations": [],
            },
            "progress": [],
        }
        self.write_progress()

    def _atomic_write(self, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_jsonable(self.payload), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def write_progress(self) -> None:
        self.payload["updated_at"] = _utc_now()
        self._atomic_write(self.progress_path)
        if hasattr(self, "document"):
            self.document["training_counts"] = {
                "physics": self.payload["counts"]["physical_steps"],
                "operator": self.payload["counts"]["fno_steps"],
                "supervised": self.payload["counts"]["warmup_steps"],
            }
            self.document["results"] = [
                {"coefficients": [row["nu"], row["rho"]],
                 "reference_coefficients": [TARGET_NU, FIXED_RHO],
                 "phase": row["phase"], "checkpoint": row["checkpoint"],
                 "relative_l2": row["target_reference_relative_l2"],
                 "evaluation_size": self.document["config"]["evaluation_size"]}
                for row in self.payload["progress"]
                if row.get("event") == "physical_block_complete"
            ]
            write_json(self.run_dir / "metrics.json", self.document)

    def add_time(self, key: str, seconds: float) -> None:
        self.payload["timing"][key] += float(seconds)

    def add_count(self, key: str, count: int) -> None:
        self.payload["counts"][key] += int(count)

    def training_elapsed(self) -> float:
        timing = self.payload["timing"]
        return float(
            timing["pinn_parameter_initialization_seconds"]
            + timing["physics_optimizer_initialization_seconds"]
            + timing["initial_support_physics_seconds"]
            + timing["curriculum_or_target_physics_seconds"]
            + timing["fno_training_seconds"]
            + timing["fno_inference_seconds"]
            + timing["mask_seconds"]
            + timing["supervised_warmup_seconds"]
            + timing["operator_target_generation_seconds"]
        )

    def finalize(self, status: str, elapsed: float, **extra: Any) -> None:
        timing = self.payload["timing"]
        timing["active_training_seconds"] = self.training_elapsed()
        timing["end_to_end_seconds"] = float(elapsed)
        self.payload.update(extra)
        self.payload["status"] = status
        self.payload["finished_at"] = _utc_now()
        self.write_progress()
        self._atomic_write(self.final_path)


class ReactionRuntime:
    def __init__(self, config: ResolvedConfig, recorder: JsonRecorder) -> None:
        self.config = config
        self.recorder = recorder

        self.grid_x, self.grid_t = self._operator_grid()
        self.grid_coords = jnp.stack([self.grid_x, self.grid_t], axis=-1)
        self.grid_points = self.grid_coords.reshape(-1, 2)
        self.fno_grid_batch = self.grid_coords[None, ...]

        self.mlp = self._make_mlp()
        self.fno = self._make_fno()

        physics_schedule = optax.exponential_decay(
            PHYSICS_INIT_LR,
            PHYSICS_TRANSITION_STEPS,
            PHYSICS_DECAY_RATE,
            staircase=False,
        )
        self.physics_optimizer = optax.adamw(
            physics_schedule,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.0,
        )
        warmup_schedule = optax.exponential_decay(
            WARMUP_INIT_LR,
            WARMUP_TRANSITION_STEPS,
            WARMUP_DECAY_RATE,
            staircase=False,
        )
        self.warmup_optimizer = optax.adamw(
            warmup_schedule,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.0,
        )
        fno_schedule = optax.exponential_decay(
            FNO_INIT_LR,
            FNO_TRANSITION_STEPS,
            FNO_DECAY_RATE,
            staircase=False,
        )
        self.fno_optimizer = optax.adamw(
            fno_schedule,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.0,
        )

        self.physics_step = self._make_physics_step()
        self.warmup_step = self._make_warmup_step()
        self.fno_step = self._make_fno_step()
        self.predict_points = jax.jit(
            lambda params, points: jnp.squeeze(
                self.mlp.apply(params, points), axis=-1
            )
        )
        self.fno_predict = jax.jit(
            lambda params, nu: jnp.squeeze(
                self.fno.apply(
                    params,
                    self.fno_grid_batch,
                    jnp.asarray([[nu, FIXED_RHO]], dtype=jnp.float32),
                )
            )
        )
        self.select_mask = self._make_mask_function()

        reference_start = time.perf_counter()
        self.reference_points_np, self.reference_values_np = (
            self._generate_reference(config.evaluation_size)
        )
        self.recorder.payload["timing_details"][
            "reference_generation_seconds"
        ] = time.perf_counter() - reference_start
        self.evaluation_chunk_size = min(
            EVALUATION_CHUNK_SIZE, len(self.reference_values_np)
        )
        self.fno_initial_params = None


    @staticmethod
    def _operator_grid() -> tuple[jax.Array, jax.Array]:
        x = jnp.linspace(X_MIN, X_MAX, X_SIZE)
        t = jnp.linspace(T_MIN, T_MAX, T_SIZE)
        return jnp.meshgrid(x, t, indexing="ij")


    @staticmethod
    def _make_mlp() -> hk.Transformed:
        def forward(inputs: jax.Array) -> jax.Array:
            initializer = hk.initializers.VarianceScaling(
                scale=2.0,
                mode="fan_in",
                distribution="truncated_normal",
            )
            hidden = inputs
            for _ in range(4):
                hidden = hk.Linear(64, w_init=initializer)(hidden)
                hidden = jax.nn.tanh(hidden)
            return hk.Linear(1, w_init=initializer)(hidden)

        return hk.without_apply_rng(hk.transform(forward))


    @staticmethod
    def _make_fno() -> hk.Transformed:
        def forward(grid: jax.Array, parameters: jax.Array) -> jax.Array:
            embedding = hk.Sequential(
                [hk.Linear(128), jax.nn.gelu, hk.Linear(32)]
            )(parameters)
            operator = FNO2D(
                modes=16,
                channels=64,
                depth=8,
                beta_emb_dim=32,
            )
            return operator(grid, embedding)

        return hk.without_apply_rng(hk.transform(forward))


    def _role_key(self, role: int, index: int) -> jax.Array:
        key = jax.random.fold_in(
            jax.random.PRNGKey(self.config.seed), int(role)
        )
        return jax.random.fold_in(key, int(index))


    def init_mlp(
        self,
        seed_offset: int,
        role: int = RNG_ROLE_MLP_INITIALIZATION,
    ) -> Any:
        key = self._role_key(role, seed_offset)
        return self.mlp.init(key, jnp.ones((1, 2), dtype=jnp.float32))


    def timed_init_mlp(self, seed_offset: int) -> Any:
        start = time.perf_counter()
        params = self.init_mlp(seed_offset)
        _sync_tree(params)
        self.recorder.add_time(
            "pinn_parameter_initialization_seconds",
            time.perf_counter() - start,
        )
        return params


    def init_physics_optimizer_state(self, params: Any) -> Any:
        start = time.perf_counter()
        state = self.physics_optimizer.init(params)
        _sync_tree(state)
        self.recorder.add_time(
            "physics_optimizer_initialization_seconds",
            time.perf_counter() - start,
        )
        return state


    def _make_physics_step(self):
        apply = self.mlp.apply
        optimizer = self.physics_optimizer

        def scalar(params: Any, point: jax.Array) -> jax.Array:
            return jnp.squeeze(apply(params, point[None, :]))

        def residual_loss(
            params: Any, nu: jax.Array, rng: jax.Array
        ) -> jax.Array:
            x_key, t_key = jax.random.split(rng)
            x = jax.random.uniform(
                x_key,
                (NUM_RESIDUAL_POINTS,),
                minval=X_MIN,
                maxval=X_MAX,
            )
            t = jax.random.uniform(
                t_key,
                (NUM_RESIDUAL_POINTS,),
                minval=T_MIN,
                maxval=T_MAX,
            )
            points = jnp.stack([x, t], axis=1)

            def residual_at(point: jax.Array) -> jax.Array:
                function = lambda coordinate: scalar(params, coordinate)
                value = function(point)
                gradient = jax.grad(function)(point)
                hessian = jax.hessian(function)(point)
                return (
                    gradient[1]
                    - nu * hessian[0, 0]
                    - FIXED_RHO * value * (1.0 - value)
                )

            residual = jax.vmap(residual_at)(points)
            return jnp.mean(jnp.square(residual))

        def initial_loss(params: Any, rng: jax.Array) -> jax.Array:
            x = jax.random.uniform(
                rng,
                (NUM_INITIAL_POINTS,),
                minval=X_MIN,
                maxval=X_MAX,
            )
            points = jnp.stack([x, jnp.zeros_like(x)], axis=1)
            target = jnp.sin(jnp.pi * x)
            prediction = jnp.squeeze(apply(params, points), axis=-1)
            return jnp.mean(jnp.square(prediction - target))

        def boundary_loss(params: Any, rng: jax.Array) -> jax.Array:
            times = jax.random.uniform(
                rng,
                (NUM_BOUNDARY_POINTS // 2,),
                minval=T_MIN,
                maxval=T_MAX,
            )
            left = jnp.stack(
                [jnp.full_like(times, X_MIN), times], axis=1
            )
            right = jnp.stack(
                [jnp.full_like(times, X_MAX), times], axis=1
            )
            points = jnp.concatenate([left, right], axis=0)
            prediction = jnp.squeeze(apply(params, points), axis=-1)
            return jnp.mean(jnp.square(prediction))

        @jax.jit
        def step(
            params: Any,
            optimizer_state: Any,
            rng: jax.Array,
            nu: jax.Array,
        ):
            residual_key, initial_key, boundary_key = jax.random.split(rng, 3)
            loss_r, gradient_r = jax.value_and_grad(residual_loss)(
                params, nu, residual_key
            )
            loss_i, gradient_i = jax.value_and_grad(initial_loss)(
                params, initial_key
            )
            loss_b, gradient_b = jax.value_and_grad(boundary_loss)(
                params, boundary_key
            )
            gradient = jax.tree_util.tree_map(
                lambda residual, initial, boundary: (
                    residual + initial + boundary
                ),
                gradient_r,
                gradient_i,
                gradient_b,
            )
            updates, optimizer_state = optimizer.update(
                gradient, optimizer_state, params
            )
            params = optax.apply_updates(params, updates)
            return params, optimizer_state, loss_r, loss_i, loss_b

        return step


    def _make_warmup_step(self):
        apply = self.mlp.apply
        optimizer = self.warmup_optimizer

        @jax.jit
        def step(
            params: Any,
            optimizer_state: Any,
            points: jax.Array,
            values: jax.Array,
        ):
            def loss_function(candidate: Any) -> jax.Array:
                prediction = jnp.squeeze(apply(candidate, points), axis=-1)
                return jnp.mean(jnp.square(prediction - values))

            loss, gradient = jax.value_and_grad(loss_function)(params)
            updates, optimizer_state = optimizer.update(
                gradient, optimizer_state, params
            )
            params = optax.apply_updates(params, updates)
            return params, optimizer_state, loss

        return step


    def _make_fno_step(self):
        apply = self.fno.apply
        optimizer = self.fno_optimizer
        base_grid = self.fno_grid_batch

        @jax.jit
        def step(
            params: Any,
            optimizer_state: Any,
            targets: jax.Array,
            parameter_pairs: jax.Array,
        ):
            def loss_function(candidate: Any) -> jax.Array:
                batch_size = targets.shape[0]
                grids = jnp.broadcast_to(
                    base_grid, (batch_size, X_SIZE, T_SIZE, 2)
                )
                prediction = jnp.squeeze(
                    apply(candidate, grids, parameter_pairs), axis=-1
                )
                return jnp.mean(jnp.square(prediction - targets))

            loss, gradient = jax.value_and_grad(loss_function)(params)
            updates, optimizer_state = optimizer.update(
                gradient, optimizer_state, params
            )
            params = optax.apply_updates(params, updates)
            return params, optimizer_state, loss

        return step


    def _make_mask_function(self):
        coordinates = self.grid_coords
        flat_coordinates = coordinates.reshape(-1, 2)
        dx = coordinates[1, 0, 0] - coordinates[0, 0, 0]
        dt = coordinates[0, 1, 1] - coordinates[0, 0, 1]

        @jax.jit
        def select(field: jax.Array, nu: jax.Array):
            u_x = jnp.gradient(field, dx, axis=0)
            u_t = jnp.gradient(field, dt, axis=1)
            u_xx = jnp.gradient(u_x, dx, axis=0)
            residual = (
                u_t
                - nu * u_xx
                - FIXED_RHO * field * (1.0 - field)
            )
            absolute = jnp.abs(residual).reshape(-1)
            indices = jnp.argsort(absolute)[:MASK_KEEP_POINTS]
            flat_field = field.reshape(-1)
            return (
                flat_coordinates[indices],
                flat_field[indices],
                absolute[indices[-1]],
                jnp.max(absolute),
            )

        return select


    @staticmethod
    def _generate_reference(size: int) -> tuple[np.ndarray, np.ndarray]:
        """Strang-split zero-Dirichlet reference on a square x/t grid."""
        x = np.linspace(X_MIN, X_MAX, size, dtype=np.float64)
        t = np.linspace(T_MIN, T_MAX, size, dtype=np.float64)
        dt_value = float(t[1] - t[0])
        current = np.sin(np.pi * x)
        current[0] = 0.0
        current[-1] = 0.0
        solution = np.empty((size, size), dtype=np.float32)
        solution[:, 0] = current.astype(np.float32)
        mode = np.arange(1, size - 1, dtype=np.float64)
        diffusion = np.exp(
            -TARGET_NU * np.square(np.pi * mode) * dt_value
        )
        reaction_half = np.exp(-FIXED_RHO * dt_value / 2.0)

        def reaction(values: np.ndarray) -> np.ndarray:
            denominator = values + (1.0 - values) * reaction_half
            output = values / denominator
            output[0] = 0.0
            output[-1] = 0.0
            return np.clip(output, 0.0, 1.0)

        for time_index in range(1, size):
            current = reaction(current)
            coefficients = dst(current[1:-1], type=1, norm="ortho")
            current[1:-1] = idst(
                coefficients * diffusion, type=1, norm="ortho"
            )
            current = reaction(current)
            solution[:, time_index] = current.astype(np.float32)

        x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
        points = np.stack(
            [x_grid.reshape(-1), t_grid.reshape(-1)], axis=1
        ).astype(np.float32)
        return points, solution.reshape(-1)


    def compile_required(self) -> float:
        start = time.perf_counter()
        dummy_params = self.init_mlp(0, role=RNG_ROLE_COMPILE_ONLY)
        dummy_state = self.physics_optimizer.init(dummy_params)
        output = self.physics_step(
            dummy_params,
            dummy_state,
            self._role_key(RNG_ROLE_COMPILE_ONLY, 1),
            jnp.asarray(TARGET_NU, dtype=jnp.float32),
        )
        _sync_tree(output)
        grid_output = self.predict_points(dummy_params, self.grid_points)
        _sync_tree(grid_output)

        eval_dummy = jnp.asarray(
            self.reference_points_np[: self.evaluation_chunk_size]
        )
        eval_output = self.predict_points(dummy_params, eval_dummy)
        _sync_tree(eval_output)

        dummy_points = self.grid_points[:MASK_KEEP_POINTS]
        dummy_values = jnp.zeros((MASK_KEEP_POINTS,), dtype=jnp.float32)
        warmup_state = self.warmup_optimizer.init(dummy_params)
        warmup_output = self.warmup_step(
            dummy_params, warmup_state, dummy_points, dummy_values
        )
        _sync_tree(warmup_output)

        key = self._role_key(RNG_ROLE_FNO_INITIALIZATION, 0)
        self.fno_initial_params = self.fno.init(
            key,
            jnp.ones((1, X_SIZE, T_SIZE, 2), dtype=jnp.float32),
            jnp.ones((1, 2), dtype=jnp.float32),
        )
        fno_state = self.fno_optimizer.init(self.fno_initial_params)
        batch_sizes = (10, 15, 16)
        for batch_size in batch_sizes:
            fno_output = self.fno_step(
                self.fno_initial_params,
                fno_state,
                jnp.zeros(
                    (batch_size, X_SIZE, T_SIZE), dtype=jnp.float32
                ),
                jnp.zeros((batch_size, 2), dtype=jnp.float32),
            )
            _sync_tree(fno_output)
        prediction = self.fno_predict(
            self.fno_initial_params,
            jnp.asarray(TARGET_NU, dtype=jnp.float32),
        )
        selected = self.select_mask(
            prediction, jnp.asarray(TARGET_NU, dtype=jnp.float32)
        )
        _sync_tree(selected)

        seconds = time.perf_counter() - start
        self.recorder.payload["timing"]["compile_seconds"] = seconds
        self.recorder.write_progress()
        return seconds


    def _training_key(self, global_step: int) -> jax.Array:
        return self._role_key(
            RNG_ROLE_PHYSICS_SAMPLING, int(global_step)
        )


    def train_physics_segment(
        self,
        params: Any,
        optimizer_state: Any,
        nu: float,
        steps: int,
        global_step_start: int,
        timing_key: str,
        block_index: int,
        structural_new_pinn: bool,
        optimizer_state_continues_after_segment: bool,
    ) -> tuple[Any, Any, dict[str, Any]]:
        _sync_tree((params, optimizer_state))
        start = time.perf_counter()
        loss_r = jnp.asarray(np.nan, dtype=jnp.float32)
        loss_i = jnp.asarray(np.nan, dtype=jnp.float32)
        loss_b = jnp.asarray(np.nan, dtype=jnp.float32)
        nu_array = jnp.asarray(nu, dtype=jnp.float32)
        for local_step in range(steps):
            (
                params,
                optimizer_state,
                loss_r,
                loss_i,
                loss_b,
            ) = self.physics_step(
                params,
                optimizer_state,
                self._training_key(global_step_start + local_step),
                nu_array,
            )
        _sync_tree((params, optimizer_state, loss_r, loss_i, loss_b))
        seconds = time.perf_counter() - start
        self.recorder.add_time(timing_key, seconds)
        self.recorder.add_count("physical_steps", steps)
        detail = {
            "block_index": block_index,
            "nu": nu,
            "rho": FIXED_RHO,
            "steps": steps,
            "seconds": seconds,
            "last_residual_loss": _float(loss_r),
            "last_initial_loss": _float(loss_i),
            "last_boundary_loss": _float(loss_b),
            "structural_new_pinn": structural_new_pinn,
            "optimizer_state_continues_after_segment": (
                optimizer_state_continues_after_segment
            ),
        }
        self.recorder.payload["timing_details"]["physical_blocks"].append(
            detail
        )
        self.latest_params = params
        return params, optimizer_state, detail


    def predict_grid(self, params: Any) -> jax.Array:
        _sync_tree(params)
        start = time.perf_counter()
        output = self.predict_points(params, self.grid_points).reshape(
            X_SIZE, T_SIZE
        )
        _sync_tree(output)
        seconds = time.perf_counter() - start
        self.recorder.add_time(
            "operator_target_generation_seconds", seconds
        )
        self.recorder.payload["timing_details"][
            "grid_generation_seconds"
        ] += seconds
        return output


    def evaluate_target(
        self, params: Any, block_index: int, nu_trained: float
    ) -> float:
        _sync_tree(params)
        start = time.perf_counter()
        squared_error = 0.0
        squared_reference = 0.0
        total = len(self.reference_values_np)
        chunk = self.evaluation_chunk_size
        for offset in range(0, total, chunk):
            stop = min(offset + chunk, total)
            points_np = self.reference_points_np[offset:stop]
            reference_np = self.reference_values_np[offset:stop]
            valid = stop - offset
            if valid < chunk:
                points_np = np.pad(
                    points_np, ((0, chunk - valid), (0, 0)), mode="edge"
                )
                reference_np = np.pad(
                    reference_np, (0, chunk - valid), mode="edge"
                )
            prediction = self.predict_points(
                params, jnp.asarray(points_np, dtype=jnp.float32)
            )
            difference = prediction[:valid] - jnp.asarray(
                reference_np[:valid], dtype=jnp.float32
            )
            squared_error += _float(jnp.sum(jnp.square(difference)))
            squared_reference += float(
                np.sum(np.square(reference_np[:valid], dtype=np.float64))
            )
        value = float(np.sqrt(squared_error / (squared_reference + 1e-24)))
        seconds = time.perf_counter() - start
        self.recorder.add_time("evaluation_seconds", seconds)
        self.recorder.payload["timing_details"]["evaluations"].append(
            {
                "block_index": block_index,
                "nu_trained": nu_trained,
                "target_nu": TARGET_NU,
                "relative_l2": value,
                "seconds": seconds,
            }
        )
        return value


    def record_block(
        self, detail: dict[str, Any], error: float, phase: str
    ) -> None:
        filename = f"checkpoints/pinn_{detail['block_index']:02d}.npz"
        save_params(self.recorder.run_dir / filename, self.latest_params)
        print(f"RD {phase}: nu={detail['nu']:.6g}, target RelL2={error:.9g}", flush=True)
        self.recorder.payload["progress"].append(
            {
                "event": "physical_block_complete",
                "phase": phase,
                "checkpoint": filename,
                **detail,
                "target_reference_relative_l2": error,
                "cumulative_counts": dict(self.recorder.payload["counts"]),
                "cumulative_active_training_seconds": (
                    self.recorder.training_elapsed()
                ),
            }
        )
        self.recorder.write_progress()


    def train_fno_stage(
        self,
        fno_params: Any,
        targets: list[jax.Array],
        nus: list[float],
        stage_index: int,
    ) -> Any:
        _sync_tree((fno_params, targets))
        start = time.perf_counter()
        all_targets = jnp.stack(targets, axis=0)
        all_pairs = jnp.asarray(
            [[nu, FIXED_RHO] for nu in nus], dtype=jnp.float32
        )
        state = self.fno_optimizer.init(fno_params)
        loss = jnp.asarray(np.nan, dtype=jnp.float32)
        for step_index in range(self.config.fno_steps_per_stage):
            if len(targets) <= 16:
                batch_targets = all_targets
                batch_pairs = all_pairs
            else:
                key = self._role_key(
                    RNG_ROLE_FNO_BATCH_SAMPLING,
                    stage_index * max(self.config.fno_steps_per_stage, 1)
                    + step_index,
                )
                indices = jax.random.choice(
                    key, len(targets), shape=(16,), replace=False
                )
                batch_targets = all_targets[indices]
                batch_pairs = all_pairs[indices]
            fno_params, state, loss = self.fno_step(
                fno_params, state, batch_targets, batch_pairs
            )
        _sync_tree((fno_params, state, loss))
        seconds = time.perf_counter() - start
        self.recorder.add_time("fno_training_seconds", seconds)
        self.recorder.add_count(
            "fno_steps", self.config.fno_steps_per_stage
        )
        detail = {
            "stage_index": stage_index,
            "training_examples": len(targets),
            "steps": self.config.fno_steps_per_stage,
            "seconds": seconds,
            "last_loss": _float(loss),
            "optimizer_state_reset_for_macro_stage": True,
            "fno_parameters_inherited": stage_index > 0,
        }
        self.recorder.payload["timing_details"]["fno_stages"].append(detail)
        self.recorder.payload["progress"].append(
            {
                "event": "fno_stage_complete",
                **detail,
                "cumulative_counts": dict(self.recorder.payload["counts"]),
            }
        )
        self.recorder.write_progress()
        return fno_params


    def operator_prediction_and_mask(
        self, fno_params: Any, nu: float, target_index: int
    ) -> tuple[jax.Array, jax.Array]:
        _sync_tree(fno_params)
        start = time.perf_counter()
        prediction = self.fno_predict(
            fno_params, jnp.asarray(nu, dtype=jnp.float32)
        )
        _sync_tree(prediction)
        inference_seconds = time.perf_counter() - start
        self.recorder.add_time("fno_inference_seconds", inference_seconds)
        self.recorder.payload["timing_details"]["fno_inference"].append(
            {
                "target_index": target_index,
                "nu": nu,
                "seconds": inference_seconds,
            }
        )

        start = time.perf_counter()
        points, values, threshold, maximum = self.select_mask(
            prediction, jnp.asarray(nu, dtype=jnp.float32)
        )
        _sync_tree((points, values, threshold, maximum))
        mask_seconds = time.perf_counter() - start
        self.recorder.add_time("mask_seconds", mask_seconds)
        self.recorder.payload["timing_details"]["masks"].append(
            {
                "target_index": target_index,
                "nu": nu,
                "seconds": mask_seconds,
                "kept_points": MASK_KEEP_POINTS,
                "total_points": X_SIZE * T_SIZE,
                "threshold": _float(threshold),
                "max_absolute_residual": _float(maximum),
            }
        )
        self.recorder.write_progress()
        return points, values


    def supervised_warmup(
        self,
        points: jax.Array,
        values: jax.Array,
        seed_offset: int,
        nu: float,
        target_index: int,
    ) -> Any:
        _sync_tree((points, values))
        params = self.timed_init_mlp(seed_offset)
        start = time.perf_counter()
        state = self.warmup_optimizer.init(params)
        loss = jnp.asarray(np.nan, dtype=jnp.float32)
        for _ in range(self.config.warmup_steps_per_target):
            params, state, loss = self.warmup_step(
                params, state, points, values
            )
        _sync_tree((params, state, loss))
        seconds = time.perf_counter() - start
        self.recorder.add_time("supervised_warmup_seconds", seconds)
        self.recorder.add_count(
            "warmup_steps", self.config.warmup_steps_per_target
        )
        self.recorder.payload["timing_details"]["warmups"].append(
            {
                "target_index": target_index,
                "nu": nu,
                "steps": self.config.warmup_steps_per_target,
                "seconds": seconds,
                "last_loss": _float(loss),
            }
        )
        self.recorder.write_progress()
        return params


    def train_support_set(
        self,
    ) -> tuple[list[jax.Array], list[float], int]:
        targets: list[jax.Array] = []
        nus = initial_nus()
        global_step = 0
        for block_index, nu in enumerate(nus):
            params = self.timed_init_mlp(10_000 + block_index)
            state = self.init_physics_optimizer_state(params)
            params, state, detail = self.train_physics_segment(
                params,
                state,
                nu,
                self.config.physics_steps_per_block,
                global_step,
                "initial_support_physics_seconds",
                block_index,
                structural_new_pinn=True,
                optimizer_state_continues_after_segment=False,
            )
            del state
            global_step += self.config.physics_steps_per_block
            error = self.evaluate_target(params, block_index, nu)
            self.record_block(detail, error, "operator_support")
            targets.append(self.predict_grid(params))
        return targets, nus, global_step


    def run_nocl(self) -> Any:
        if self.fno_initial_params is None:
            raise RuntimeError("FNO was not initialized during compilation")
        targets, nus, global_step = self.train_support_set()
        fno_params = self.fno_initial_params
        final_params = None
        target_index = 0
        block_index = INITIAL_SUPPORT_COUNT
        for stage_index, (low, high, count) in enumerate(CURRICULUM_STAGES):
            fno_params = self.train_fno_stage(
                fno_params, targets, nus, stage_index
            )
            for nu in stage_nus(low, high, count):
                points, values = self.operator_prediction_and_mask(
                    fno_params, nu, target_index
                )
                params = self.supervised_warmup(
                    points,
                    values,
                    20_000 + target_index,
                    nu,
                    target_index,
                )
                state = self.init_physics_optimizer_state(params)
                params, state, detail = self.train_physics_segment(
                    params,
                    state,
                    nu,
                    self.config.physics_steps_per_block,
                    global_step,
                    "curriculum_or_target_physics_seconds",
                    block_index,
                    structural_new_pinn=True,
                    optimizer_state_continues_after_segment=False,
                )
                del state
                global_step += self.config.physics_steps_per_block
                error = self.evaluate_target(params, block_index, nu)
                self.record_block(detail, error, "operator_curriculum")
                targets.append(self.predict_grid(params))
                nus.append(nu)
                final_params = params
                target_index += 1
                block_index += 1
        if final_params is None:
            raise RuntimeError("NOCL produced no target PINN")
        self.final_fno_params = fno_params
        return final_params



def evaluate_checkpoint(params, size=1280):
    """Use the original padded/chunked metric and float64 reference norm."""
    model = ReactionRuntime._make_mlp()
    predict = jax.jit(lambda weights, points: jnp.squeeze(
        model.apply(weights, points), axis=-1))
    points, reference = ReactionRuntime._generate_reference(size)
    chunk = min(EVALUATION_CHUNK_SIZE, len(reference))
    squared_error = 0.0
    squared_reference = 0.0
    pieces = []
    for offset in range(0, len(reference), chunk):
        stop = min(offset + chunk, len(reference))
        points_np = points[offset:stop]
        reference_np = reference[offset:stop]
        valid = stop - offset
        if valid < chunk:
            points_np = np.pad(points_np, ((0, chunk-valid), (0, 0)), mode="edge")
            reference_np = np.pad(reference_np, (0, chunk-valid), mode="edge")
        prediction = predict(params, jnp.asarray(points_np, dtype=jnp.float32))
        difference = prediction[:valid] - jnp.asarray(reference_np[:valid], dtype=jnp.float32)
        squared_error += _float(jnp.sum(jnp.square(difference)))
        squared_reference += float(np.sum(np.square(reference_np[:valid], dtype=np.float64)))
        pieces.append(np.asarray(prediction[:valid]))
    error = float(np.sqrt(squared_error / (squared_reference + 1e-24)))
    return error, np.concatenate(pieces).reshape(size, size), reference.reshape(size, size)


class Experiment:
    def __init__(self, config, output, save_operator=False):
        self.start = time.perf_counter()
        self.config = resolve_config(config)
        self.output = Path(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"Choose an empty output directory: {self.output}")
        (self.output / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.save_operator = save_operator
        config = dict(config)
        config.update(vars(self.config))
        protocol = json.loads(files("experiments").joinpath(
            "records/rd_protocol.json").read_text())
        protocol["seed"] = self.config.seed
        protocol["mode"] = "smoke" if self.config.smoke else "full"
        protocol["scientifically_valid"] = not self.config.smoke
        protocol["evaluation"]["grid"] = [self.config.evaluation_size] * 2
        protocol["executed_counts"] = {
            "physical_steps": TOTAL_BLOCKS * self.config.physics_steps_per_block,
            "warmup_steps": CURRICULUM_TARGET_COUNT * self.config.warmup_steps_per_target,
            "fno_steps": len(CURRICULUM_STAGES) * self.config.fno_steps_per_stage,
        }
        self.recorder = JsonRecorder(self.output, protocol, environment(), False)
        self.document = {
            "schema_version": 2, "status": "running", "problem": "rd", "method": "nocl",
            "smoke": self.config.smoke, "config": config, "environment": environment(),
            "results": [], "evaluation_reference": "fixed target (nu,rho)=(4,5)",
        }
        self.recorder.document = self.document
        self.flush()

    def flush(self):
        self.document["elapsed_seconds"] = time.perf_counter() - self.start
        self.recorder.write_progress()

    def run(self):
        runtime = ReactionRuntime(self.config, self.recorder)
        seconds = runtime.compile_required()
        print(f"RD compilation: {seconds:.2f}s; starting NOCL", flush=True)
        params = runtime.run_nocl()
        error = runtime.evaluate_target(params, TOTAL_BLOCKS, TARGET_NU)
        if not np.isfinite(error):
            raise FloatingPointError("Non-finite RD metric")
        if self.recorder.payload["counts"] != self.recorder.payload["protocol"]["executed_counts"]:
            raise AssertionError("Training step counts do not match the protocol")
        self.document.update(status="completed", final_relative_l2=error,
                             final_checkpoint=f"checkpoints/pinn_{TOTAL_BLOCKS-1:02d}.npz")
        if self.save_operator:
            save_params(self.output / "operator.npz", runtime.final_fno_params)
        self.recorder.finalize("completed", time.perf_counter()-self.start,
                              final_metrics={"target_relative_l2": error,
                                             "target_nu": TARGET_NU, "target_rho": FIXED_RHO})
        self.flush()
        return self.document
