# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Mixin adding container support to a workload."""

from production_test_framework.docker import (
    force_remove_container,
    label_args,
    unique_container_name,
)

__all__ = ["DockerContainerMixin"]


class DockerContainerMixin:
    """
    Adds "runs its work in a container" to a workload, without changing what kind it is.

    Mix in ahead of the workload base so its :meth:`_cleanup_after_run` takes precedence
    """

    container_name_prefix = "workload"

    def __init__(
        self,
        *,
        image_name: str | None = None,
        container_name: str | None = None,
        env: dict[str, str] | None = None,
        docker_extra_args: tuple[str, ...] = (),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._image_name = image_name
        self._container_name = container_name or unique_container_name(self.container_name_prefix)
        self._env = dict(env or {})
        self._docker_extra_args = tuple(docker_extra_args)

    @property
    def image_name(self) -> str | None:
        return self._image_name

    @property
    def container_name(self) -> str:
        """The container this run uses, generated per instance unless one was supplied."""
        return self._container_name

    @property
    def uses_docker(self) -> bool:
        """
        Whether this run actually starts a container.
        """
        return True

    def env_args(self, flag: str = "-e") -> list[str]:
        """
        Environment arguments, ``-e KEY=VALUE`` by default.
        """
        args: list[str] = []
        for key, value in self._env.items():
            args.extend([flag, f"{key}={value}"])
        return args

    def docker_run_argv(self, *flags: str) -> list[str]:
        """
        The ``docker run`` portion of a command, up to and including the image.
        """
        return [
            "docker",
            "run",
            "--rm",
            "-t",
            *flags,
            "--name",
            self._container_name,
            *label_args(self._container_name),
            *self.env_args(),
            *self._docker_extra_args,
            *([self._image_name] if self._image_name else []),
        ]

    def _cleanup_after_run(self) -> None:
        """
        Remove the container, which ``--rm`` does not do for a stopped or timed-out run.
        """
        if self.uses_docker:
            force_remove_container(self._container_name)
